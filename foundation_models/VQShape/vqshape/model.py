import torch
import torch.nn as nn
import torch.distributions as D
from einops import rearrange, repeat
from vqshape.networks import MLP, ShapeDecoder


# Utility functions
def log(t, eps = 1e-5):
    return t.clamp(min = eps).log()

def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)

def onehot_straight_through(p: torch.Tensor):
    max_idx = p.argmax(-1)
    onehot = nn.functional.one_hot(max_idx, p.shape[-1])
    return onehot + p - p.detach()


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_length, embedding_dim):
        super().__init__()
        self.positional_embeddings = nn.Embedding(max_seq_length, embedding_dim)

    def forward(self, x):
        # Create a tensor of positional indices: [0, 1, 2, ..., max_seq_length-1]
        position_indices = torch.arange(0, x.size(1)).long().unsqueeze(0).to(x.device)
        
        # Retrieve the positional embeddings corresponding to the indices
        pos_embeddings = self.positional_embeddings(position_indices)
        
        return pos_embeddings + x


class EuclCodebook(nn.Module):
    def __init__(
            self, 
            num_code: int = 512, 
            dim_code: int = 256, 
            commit_loss=1., 
            entropy_loss=0., 
            entropy_gamma=1.,
        ):
        super().__init__()
        self.num_codebook_vectors = num_code
        self.latent_dim = dim_code
        self.commit_loss = commit_loss
        self.entropy_loss = entropy_loss
        self.entropy_gamma = entropy_gamma
        
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        z_flattened = rearrange(z, "B L E -> (B L) E")

        # Compute distance between z and codebook vectors
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))

        # Find the nearest codebook vector for each z
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = rearrange(self.embedding(min_encoding_indices), '(B L) E -> B L E', B=z.shape[0])

        # Commitment loss
        loss = torch.mean((z_q.detach() - z)**2) + self.commit_loss * torch.mean((z_q - z.detach())**2)
        z_q = z + (z_q - z).detach()
        min_encoding_indices = rearrange(min_encoding_indices, '(B L) -> B L', B=z.shape[0])

        # Entropy loss
        if self.entropy_loss > 0:
            p = nn.functional.softmin(d/0.01, dim=-1)
            entropy_loss = entropy(p).mean() - self.entropy_gamma * entropy(p.mean(0))
            loss += self.entropy_loss * entropy_loss

        return z_q, min_encoding_indices, loss


class PatchEncoder(nn.Module):
    def __init__(
            self,
            dim_embedding: int = 256,
            patch_size: int = 8,
            num_patch: int = 64,
            num_head: int = 6,
            num_layer: int = 6,
        ):
        super().__init__()

        self.patch_size = patch_size
        self.num_patch = num_patch

        # Embedding Layers
        self.pos_embed = PositionalEmbedding(num_patch, dim_embedding)
        self.input_project = nn.Linear(patch_size, dim_embedding)
        
        # Transformer Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embedding,
            nhead=num_head,
            dropout=0.1,
            dim_feedforward=dim_embedding*4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layer
        )

    def patch_and_embed(self, x):
        x = x.unfold(-1, self.patch_size, int(x.shape[-1]/self.num_patch))
        x = self.pos_embed(self.input_project(x))
        return x

    def forward(self, x):
        return self.transformer(self.patch_and_embed(x))
    

class PatchDecoder(nn.Module):
    def __init__(
            self, 
            dim_embedding: int = 256,
            patch_size: int = 8,
            num_head: int = 6,
            num_layer: int = 6,
        ):
        super().__init__()

        self.patch_size = patch_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embedding,
            nhead=num_head,
            dropout=0.1,
            dim_feedforward=dim_embedding*4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layer
        )
        self.out_layer = nn.Linear(dim_embedding, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_embedding)/dim_embedding)

    def forward(self, x):
        x = torch.cat([repeat(self.cls_token, '1 1 E -> B 1 E', B=x.shape[0]), x], dim=1)
        out = self.transformer(x)
        x_hat = rearrange(self.out_layer(out[:, 1:, :]), "B L E -> B (L E)")
        return x_hat, out[:, 0, :]


class Tokenizer(nn.Module):
    def __init__(
            self,
            dim_embedding: int = 256,
            num_token: int = 32,
            num_head: int = 6,
            num_layer: int = 6,
        ):
        super().__init__()

        self.tokens = nn.Parameter(torch.randn(1, num_token, dim_embedding)/dim_embedding)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_embedding,
            nhead=num_head,
            dropout=0.1,
            dim_feedforward=dim_embedding*4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layer
        )

    def forward(self, x, memory_mask=None):
        return self.transformer(repeat(self.tokens, '1 n d -> b n d', b=x.shape[0]), x, memory_key_padding_mask=memory_mask)


class AttributeDecoder(nn.Module):
    '''
    Decode embeddings into shape attributes
    '''
    def __init__(self, dim_code: int = 256, dim_embedding: int = 256) -> None:
        super().__init__()

        self.z_head = MLP(dim_embedding, dim_code, dim_embedding)
        self.tl_mean_head = MLP(dim_embedding, 2, dim_embedding)
        self.mu_head = MLP(dim_embedding, 1, dim_embedding)
        self.sigma_head = MLP(dim_embedding, 1, dim_embedding)

    def forward(self, x):
        return (
            self.z_head(x),
            nn.functional.sigmoid(self.tl_mean_head(x)),
            self.mu_head(x),
            nn.functional.softplus(self.sigma_head(x))
        )
    

class AttributeEncoder(nn.Module):
    '''
    Encode shape attributes into embeddings
    '''
    def __init__(self, dim_code: int = 256, dim_embedding: int = 256) -> None:
        super().__init__()

        self.project = nn.Linear(dim_code + 4, dim_embedding)
    
    def forward(self, z, t, l, mu, sigma):
        return self.project(torch.cat([z, mu, sigma, t, l], dim=-1))


def extract_subsequence(x, t, l, norm_length, smooth=9):
    '''
    Sample subsequences specified by t and l from time series x
    '''
    B, T = x.shape
    relative_positions = torch.linspace(0, 1, steps=norm_length).to(x.device)
    start_indices = (t * (T-1))
    end_indices = (torch.clamp(t + l, max=1) * (T-1))

    grid = start_indices + (end_indices - start_indices) * relative_positions.unsqueeze(0)
    grid = 2.0 * grid / (T - 1) - 1
    grid = torch.stack([grid, torch.ones_like(grid)], dim=-1)

    x = x.unsqueeze(1).unsqueeze(2)
    interpolated = nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return moving_average(interpolated.squeeze(1).squeeze(1), smooth)


def moving_average(x, window_size):
    B, C, _ = x.shape
    filter = torch.ones(C, 1, window_size, device=x.device) / window_size
    padding = window_size // 2 # Tuple for padding (left, right)
    x = torch.cat([torch.ones(B, C, padding, device=x.device)*x[:, :, [0]], x, torch.ones(B, C, padding, device=x.device)*x[:, :, [-1]]], dim=-1)
    smoothed_x = nn.functional.conv1d(x, filter, groups=C)
    
    return smoothed_x


def eucl_sim_loss(x, threshold=0.1):
    d = torch.norm(x.unsqueeze(1) - x.unsqueeze(2), dim=-1)
    loss = nn.functional.relu(threshold - d)
    mask = torch.ones_like(loss) - torch.eye(loss.shape[-1], device=loss.device).unsqueeze(0)
    return (loss * mask).mean()


class VQShape(nn.Module):
    def __init__(
            self, 
            dim_embedding: int = 256, # Embedding dimension of Transformers
            patch_size: int = 8, # Patch size of PatchTST backbone
            num_patch: int = 64, # Number of patches of PatchTST backbone
            num_enc_head: int = 6, # Number of heads in Transformer encoder
            num_enc_layer: int = 6, # Number of layers in Transformer encoder
            num_tokenizer_head: int = 6, # Number of heads in Transformer tokenizer
            num_tokenizer_layer: int = 6, # Number of layers in Transformer tokenizer
            num_dec_head: int = 6, # Number of heads in Transformer decoder
            num_dec_layer: int = 6, # Number of layers in Transformer decoder
            num_token: int = 32, # Number of shape tokens (output of tokenizer)
            len_s: int = 256, # Unified length of shapes
            len_input: int = 512, # Unified length of input time series
            s_smooth_factor: int = 11, # Smoothing factor for moving average
            num_code: int = 512, # Codebook size
            dim_code: int = 8, # Shape code dimension
            codebook_type: str = "standard", # Type of codebook
            lambda_commit: float = 1., # Commitment loss coefficient
            lambda_entropy: float = 1., # Entropy loss coefficient of the codebook
            entropy_gamma: float = 1., # Entropy gamma of the codebook
            mask_ratio: float = 0.25 # Mask ratio for pretraining
        ):
        super().__init__()
        
        self.len_s = len_s
        self.s_smooth_factor = s_smooth_factor
        self.num_code = num_code
        self.codebook_type = codebook_type
        self.min_shape_len = 1/64
        self.entropy_gamma = entropy_gamma

        self.num_patch = num_patch
        self.patch_size = patch_size  
        self.mask_ratio = mask_ratio
        self.num_token = num_token

        self.encoder = PatchEncoder(
            dim_embedding=dim_embedding,
            patch_size=patch_size,
            num_patch=num_patch,
            num_head=num_enc_head,
            num_layer=num_enc_layer
        )

        if codebook_type == "standard":
            self.codebook = EuclCodebook(
                num_code, 
                dim_code, 
                commit_loss=lambda_commit,
                entropy_loss=lambda_entropy,
                entropy_gamma=entropy_gamma
            )
        elif codebook_type == "vqtorch":
            from vqtorch.nn import VectorQuant
            self.codebook = VectorQuant(
                feature_size=dim_code,
                num_codes=num_code,
                beta=0.95,
                kmeans_init=False,
                affine_lr=10,
                sync_nu=0.2,
                replace_freq=40,
                dim=-1
            )
        else:
            raise NotImplementedError(f"Invalid codebook type [{codebook_type}].")
        
        self.decoder = PatchDecoder(
            dim_embedding=dim_embedding,
            patch_size=int(len_input / num_token),
            num_head=num_dec_head,
            num_layer=num_dec_layer
        )

        self.tokenizer = Tokenizer(
            dim_embedding=dim_embedding,
            num_token=num_token,
            num_head=num_tokenizer_head,
            num_layer=num_tokenizer_layer
        )

        self.attr_encoder = AttributeEncoder(dim_code=dim_code, dim_embedding=dim_embedding)
        self.attr_decoder = AttributeDecoder(dim_code=dim_code, dim_embedding=dim_embedding)
        self.shape_decoder = ShapeDecoder(dim_code, len_s, out_kernel_size=s_smooth_factor)

    def forward(self, x, *, mode='pretrain', num_input_patch=-1, mask=None, finetune=False):
        '''
        x: shape (batch_size, time_steps), time series data
        mode: mode of the forward pass
        num_input_patch: number of patches of the input time series (!! set if x is a partial time series, e.g. forecasting)
        mask: mask that indicates the missing values in the input time series (for imputation)
        finetune: whether to compute loss and update parameters for downstream tasks
        '''
        if mode == 'pretrain':
            return self.pretrain(x)
        elif mode == 'evaluate':
            return self.evaluate(x)
        elif mode == 'tokenize':
            return self.tokenize(x)
        elif mode == 'forecast':
            return self.forecast(x, num_input_patch, finetune)
        elif mode == 'imputation':
            return self.imputation(x, mask, finetune)
        else:
            raise NotImplementedError(f"VQShape: Invalid mode [{mode}]")

    def forecast(self, x: torch.Tensor, num_patch: int, finetune=False):
        self.x_mean = x[:, :num_patch*self.patch_size].mean(dim=-1, keepdims=True)
        self.x_std = (x[:, :num_patch*self.patch_size].var(dim=-1, keepdims=True) + 1e-5).sqrt()
        x_embed = self.encoder.patch_and_embed((x - self.x_mean)/self.x_std)
        x_embed = x_embed[:, :num_patch, :]
        x_embed = self.encoder.transformer(x_embed)
        # memory_attn_mask = torch.zeros(x.shape[0], x_embed.shape[1], device=x.device).bool()
        if finetune:
            _, loss_dict = self._forward(x, x_embed, None, compute_loss=True)
            return loss_dict
        else:
            output_dict = self._forward(x, x_embed, None, compute_loss=False)
            return output_dict
        
    def imputation(self, x: torch.Tensor, mask: torch.Tensor, finetune=False):
        self.x_mean = x[mask == 0].mean(dim=-1, keepdims=True)
        self.x_std = (x[mask == 0].var(dim=-1, keepdims=True) + 1e-5).sqrt()
        x_embed = self.encoder.patch_and_embed((x - self.x_mean)/self.x_std)
        mask = mask.unfold(-1, self.patch_size, int(x.shape[-1]/self.num_patch))
        mask = mask.sum(-1) > 0

        x_embed = self.encoder.transformer(x_embed, src_key_padding_mask=mask)
        if finetune:
            output_dict, loss_dict = self._forward(x, x_embed, mask, compute_loss=True)
            return output_dict['x_pred'], loss_dict
        else:
            output_dict = self._forward(x, x_embed, mask, compute_loss=False)
            return output_dict['x_pred'], output_dict
        
    def tokenize(self, x: torch.Tensor):
        self.x_mean = x.mean(dim=-1, keepdims=True)
        self.x_std = (x.var(dim=-1, keepdims=True) + 1e-5).sqrt()
        # Patch and embed the ts data
        x_embed = self.encoder((x - self.x_mean)/self.x_std)
        output_dict = self._forward(x, x_embed, None, compute_loss=False)

        # Token embedding
        tokens = torch.cat([output_dict['code'], output_dict['t_pred'], output_dict['l_pred'], output_dict['mu_pred'], output_dict['sigma_pred']], dim=-1)
        # Histogram embedding
        histogram = torch.zeros(output_dict['code'].shape[0], self.num_code, device=x.device, dtype=output_dict['code_idx'].dtype).scatter_add_(1, output_dict['code_idx'], torch.ones_like(output_dict['code_idx']))

        representations = {
            'token': tokens,
            'histogram': histogram
        }

        return representations, output_dict
    
    def evaluate(self, x: torch.Tensor):
        self.x_mean = x.mean(dim=-1, keepdims=True)
        self.x_std = (x.var(dim=-1, keepdims=True) + 1e-5).sqrt()
        # Patch and embed the ts data
        x_embed = self.encoder((x - self.x_mean)/self.x_std)
        return self._forward(x, x_embed, None, compute_loss=True)

    def pretrain(self, x: torch.Tensor):
        self.x_mean = x.mean(dim=-1, keepdims=True)
        self.x_std = (x.var(dim=-1, keepdims=True) + 1e-5).sqrt()
        # Patch and embed the ts data
        x_embed = self.encoder.patch_and_embed((x - self.x_mean)/self.x_std)
        batch_size, num_patch, patch_dim = x_embed.shape

        # Mask a subset of the patches
        num_masked = int(self.mask_ratio * num_patch)
        rand_indices = torch.rand(batch_size, num_patch, device=x.device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        x_unmasked = x_embed[torch.arange(batch_size, device=x.device).unsqueeze(1), unmasked_indices]

        # Encode unmasked patches
        x_unmasked = self.encoder.transformer(x_unmasked)

        return self._forward(x, x_unmasked, None, compute_loss=True)

    def _forward(
            self, 
            x: torch.Tensor, 
            x_embed: torch.Tensor, 
            tokenizer_attn_mask: torch.Tensor, 
            compute_loss: bool = False
        ):
        '''
        x: shape (batch_size, time_steps), time series data
        x_embed: shape (batch_size, num_patch, dim_embedding), embedded patches of the input time series
        tokenizer_attn_mask: shape (batch_size, num_token), mask that indicates the missing values in the input patches
        compute_loss: whether to compute loss
        '''

        # Tokenize
        h_shape = self.tokenizer(x_embed, tokenizer_attn_mask)

        # Latent space operations to quantize and decode the tokens
        z_e, tl_sample, mu_hat, sigma_hat = self.attr_decoder(h_shape)
        t_hat, l_hat = tl_sample[...,[0]], tl_sample[...,[1]]

        t_hat = t_hat * (1 - self.min_shape_len)
        l_hat = l_hat * (1 - t_hat) + self.min_shape_len
        if self.codebook_type == 'standard':
            z_q, z_idx, z_loss = self.codebook(z_e)
        else:
            z_q, vq_dict = self.codebook(z_e)
            z_loss = vq_dict['loss']
            z_idx = vq_dict['q']
        h_hat = self.attr_encoder(z_q, t_hat, l_hat, mu_hat, sigma_hat)

        # Reconstruct time-series
        x_hat, _ = self.decoder(h_hat)
        x_hat = x_hat * self.x_std + self.x_mean

        # Decode tokens into shapes
        s_hat_norm, _, _ = self.shape_decoder(z_q)
        s_hat = s_hat_norm * sigma_hat + mu_hat
        s_hat = s_hat * self.x_std.unsqueeze(1) + self.x_mean.unsqueeze(1)

        output_dict = {
            'x_true': x,
            'x_pred': x_hat,
            's_true': None,
            's_pred': s_hat,
            'code': z_q,
            'code_idx': z_idx,
            't_pred': t_hat,
            'l_pred': l_hat,
            'mu_pred': mu_hat,
            'sigma_pred': sigma_hat
        }

        # Compute loss
        if compute_loss:
            # Reconstruction loss
            x_loss = nn.functional.mse_loss(x_hat, x)
            s = extract_subsequence(x, t_hat, l_hat, self.len_s, smooth=self.s_smooth_factor)
            s_loss = nn.functional.mse_loss(s_hat, s.detach())
            output_dict['s_true'] = s

            # Disentanglement loss
            log_l = l_hat.log() / (torch.ones_like(l_hat) * self.min_shape_len).log()
            dist_loss = eucl_sim_loss(torch.cat([torch.cos(torch.pi * t_hat) * log_l, torch.sin(torch.pi * t_hat) * log_l], dim=-1), 0.2)

            loss_dict = {
                'ts_loss': x_loss.unsqueeze(0),
                'vq_loss': z_loss.unsqueeze(0),
                'shape_loss': s_loss.unsqueeze(0),
                'dist_loss': dist_loss.unsqueeze(0)
            }
            return output_dict, loss_dict
        else:
            return output_dict





