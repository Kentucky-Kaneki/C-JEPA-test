"""
Host-Token Representation for Cyber-JEPA.

Encodes each of the 13 network hosts as an explicit semantic entity combining host identity,
subnet membership, activity, compromise status, visibility, and relative time embeddings.
"""

from typing import Any, cast
import torch
import torch.nn as nn

from cyber_jepa.models.context import ContextTokens, TokenType


class HostTokenRepresentation(nn.Module):
    """Host-level tokenization and Transformer encoder."""

    host_subnet_ids: torch.Tensor

    def __init__(
        self,
        num_hosts: int = 13,
        num_subnets: int = 3,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        history_len: int = 4,
    ):
        super().__init__()
        self.num_hosts = num_hosts
        self.num_subnets = num_subnets
        self.hidden_dim = hidden_dim
        self.history_len = history_len

        self.host_id_emb: nn.Module = nn.Embedding(num_hosts, hidden_dim)
        self.subnet_id_emb: nn.Module = nn.Embedding(num_subnets, hidden_dim)
        self.activity_emb: nn.Module = nn.Embedding(5, hidden_dim)
        self.compromise_emb: nn.Module = nn.Embedding(5, hidden_dim)
        self.visibility_emb: nn.Module = nn.Embedding(2, hidden_dim)
        self.time_emb: nn.Module = nn.Embedding(history_len, hidden_dim)

        self.fusion_mlp: nn.Module = nn.Sequential(
            nn.Linear(hidden_dim * 5, ffn_dim),
            nn.LayerNorm(ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

        self.reg_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        host_subnets = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        self.register_buffer("host_subnet_ids", torch.tensor(host_subnets, dtype=torch.long))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def encode_context(
        self,
        flat_obs: torch.Tensor,
        host_known_mask: torch.Tensor | None = None,
        return_context_tokens: bool = True,
    ) -> ContextTokens | torch.Tensor:
        """
        Input: flat_obs [B, T_hist, 52] or [B, T_hist, 13, 4]
        Returns: ContextTokens dataclass containing tokens [B, T_hist * 13, D]
        """
        if flat_obs.dim() == 3:
            B, T_hist = flat_obs.shape[0], flat_obs.shape[1]
            x_hosts = flat_obs.view(B, T_hist, self.num_hosts, 4)
        else:
            B, T_hist = flat_obs.shape[0], flat_obs.shape[1]
            x_hosts = flat_obs

        device = flat_obs.device

        act_logits = x_hosts[:, :, :, :2].argmax(dim=-1)
        comp_logits = x_hosts[:, :, :, 2:].argmax(dim=-1)

        host_ids = torch.arange(self.num_hosts, device=device).unsqueeze(0).unsqueeze(0)
        subnet_ids = self.host_subnet_ids.unsqueeze(0).unsqueeze(0)

        e_host: torch.Tensor = self.host_id_emb(host_ids).expand(B, T_hist, -1, -1)
        e_sub: torch.Tensor = self.subnet_id_emb(subnet_ids).expand(B, T_hist, -1, -1)
        e_act: torch.Tensor = self.activity_emb(act_logits)
        e_comp: torch.Tensor = self.compromise_emb(comp_logits)

        if host_known_mask is not None:
            mask_long = host_known_mask.long()
            if mask_long.dim() == 2:
                mask_long = mask_long.unsqueeze(1).expand(-1, T_hist, -1)
            e_vis: torch.Tensor = self.visibility_emb(mask_long)
        else:
            e_vis = self.visibility_emb(torch.ones((B, T_hist, self.num_hosts), device=device, dtype=torch.long))

        cat_embs = torch.cat([e_host, e_sub, e_act, e_comp, e_vis], dim=-1)
        fused: torch.Tensor = self.fusion_mlp(cat_embs)

        t_ids_1d = torch.arange(T_hist, device=device)
        t_ids = t_ids_1d.unsqueeze(0).unsqueeze(2)
        e_time: torch.Tensor = self.time_emb(t_ids)
        tokens = fused + e_time

        seq_tokens = tokens.view(B, T_hist * self.num_hosts, self.hidden_dim)

        reg = self.reg_token.expand(B, -1, -1)
        seq_tokens = torch.cat([seq_tokens, reg], dim=1)

        out: torch.Tensor = self.transformer(seq_tokens)
        out = self.norm(out)

        global_token = out[:, -1, :]
        host_out = out[:, :-1, :]

        if not return_context_tokens:
            return host_out.view(B, T_hist, self.num_hosts, self.hidden_dim)

        total_L = T_hist * self.num_hosts
        padding_mask = torch.zeros((B, total_L), device=device, dtype=torch.bool)
        time_ids = t_ids_1d.repeat_interleave(self.num_hosts).unsqueeze(0).expand(B, -1)
        entity_ids = torch.arange(self.num_hosts, device=device).repeat(T_hist).unsqueeze(0).expand(B, -1)
        token_type_ids = torch.full((B, total_L), TokenType.HOST, device=device, dtype=torch.long)

        vis_mask = None
        if host_known_mask is not None:
            v = host_known_mask
            if v.dim() == 2:
                v = v.unsqueeze(1).expand(-1, T_hist, -1)
            vis_mask = v.reshape(B, total_L).bool()

        ctx = ContextTokens(
            tokens=host_out,
            padding_mask=padding_mask,
            visibility_mask=vis_mask,
            time_ids=time_ids,
            entity_ids=entity_ids,
            token_type_ids=token_type_ids,
            global_token=global_token,
            metadata={"history_len": T_hist, "num_hosts": self.num_hosts},
        )
        ctx.validate()
        return ctx

    def forward(
        self,
        flat_obs: torch.Tensor,
        host_known_mask: torch.Tensor | None = None,
    ) -> ContextTokens | torch.Tensor:
        return self.encode_context(flat_obs, host_known_mask=host_known_mask, return_context_tokens=True)


class InterpretableTransformerLayer(nn.Module):
    """Transformer Encoder Layer with optional self-attention weight extraction."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.last_attn_weights: torch.Tensor | None = None

    def forward(self, src: torch.Tensor, need_weights: bool = False) -> torch.Tensor:
        src_norm = self.norm1(src)
        attn_out, attn_weights = self.self_attn(
            src_norm, src_norm, src_norm, need_weights=need_weights, average_attn_weights=True
        )
        if need_weights:
            self.last_attn_weights = attn_weights.detach()
        else:
            self.last_attn_weights = None
        src = src + self.dropout1(attn_out)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
        src = src + self.dropout2(src2)
        return src


class ModernHostTokenRepresentation(nn.Module):
    """
    Modernized Host Token Representation for Cyber-JEPA.

    Supports:
    - Variable host counts N in {3..100+} with shared feature projection.
    - Continuous/Normalized telemetry input without quantization artifacts.
    - Pre-LN Interpretable Transformer with self-attention weight extraction.
    - Direct compatibility with CyberJEPA context aggregator and VICReg.
    """

    def __init__(
        self,
        num_hosts: int = 13,
        features_per_host: int = 4,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        history_len: int = 4,
    ):
        super().__init__()
        self.num_hosts = num_hosts
        self.features_per_host = features_per_host
        self.hidden_dim = hidden_dim
        self.history_len = history_len

        # Shared feature projection across all hosts
        self.host_proj = nn.Sequential(
            nn.Linear(features_per_host, ffn_dim),
            nn.LayerNorm(ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

        # Learned entity ID embedding up to max_hosts (defaults to 128)
        self.max_hosts = max(num_hosts, 128)
        self.entity_emb = nn.Embedding(self.max_hosts, hidden_dim)

        # Relative time embedding for history timesteps
        self.time_emb = nn.Embedding(history_len, hidden_dim)

        # Global registration token (acts as summary readout)
        self.reg_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Interpretable Transformer layers
        self.layers = nn.ModuleList([
            InterpretableTransformerLayer(hidden_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def encode_context(
        self,
        x: torch.Tensor,
        host_known_mask: torch.Tensor | None = None,
        return_context_tokens: bool = True,
        need_weights: bool = False,
    ) -> ContextTokens | torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1) # [B, 1, obs_dim]

        B, T_hist = x.shape[0], x.shape[1]
        device = x.device

        # x is guaranteed 3D here: [B, T_hist, obs_dim]
        N = x.shape[2] // self.features_per_host
        x_reshaped = x.view(B, T_hist, N, self.features_per_host)

        # 1. Project per-host telemetry features
        h_feats = self.host_proj(x_reshaped) # [B, T_hist, N, hidden_dim]

        # 2. Add entity & time embeddings
        entity_ids = torch.arange(N, device=device).unsqueeze(0).unsqueeze(0)
        e_ent = self.entity_emb(torch.clamp(entity_ids, 0, self.max_hosts - 1))

        t_ids = torch.clamp(torch.arange(T_hist, device=device).unsqueeze(0).unsqueeze(2), 0, self.history_len - 1)
        e_time = self.time_emb(t_ids)

        tokens = h_feats + e_ent + e_time
        seq_tokens = tokens.view(B, T_hist * N, self.hidden_dim)

        # 3. Append global registration token
        reg = self.reg_token.expand(B, -1, -1)
        seq = torch.cat([seq_tokens, reg], dim=1) # [B, T_hist * N + 1, hidden_dim]

        # 4. Forward through transformer layers
        for layer in self.layers:
            seq = layer(seq, need_weights=need_weights)
        seq = self.norm(seq)

        global_token = seq[:, -1, :]
        host_tokens = seq[:, :-1, :]

        if not return_context_tokens:
            return host_tokens.view(B, T_hist, N, self.hidden_dim)

        total_L = T_hist * N
        padding_mask = torch.zeros((B, total_L), device=device, dtype=torch.bool)
        t_ids_1d = torch.arange(T_hist, device=device)
        time_ids = t_ids_1d.repeat_interleave(N).unsqueeze(0).expand(B, -1)
        entity_ids_flat = torch.arange(N, device=device).repeat(T_hist).unsqueeze(0).expand(B, -1)
        token_type_ids = torch.full((B, total_L), TokenType.HOST, device=device, dtype=torch.long)

        vis_mask = None
        if host_known_mask is not None:
            v = host_known_mask
            if v.dim() == 2:
                v = v.unsqueeze(1).expand(-1, T_hist, -1)
            vis_mask = v.reshape(B, total_L).bool()

        ctx = ContextTokens(
            tokens=host_tokens,
            padding_mask=padding_mask,
            visibility_mask=vis_mask,
            time_ids=time_ids,
            entity_ids=entity_ids_flat,
            token_type_ids=token_type_ids,
            global_token=global_token,
            metadata={"history_len": T_hist, "num_hosts": N},
        )
        ctx.validate()
        return ctx

    def forward(
        self,
        x: torch.Tensor,
        host_known_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> ContextTokens | torch.Tensor:
        return self.encode_context(x, host_known_mask=host_known_mask, return_context_tokens=True, need_weights=need_weights)

    def get_attention_weights(self) -> list[torch.Tensor]:
        """Return list of attention weight matrices from all layers [B, L, L]."""
        return [l.last_attn_weights for l in self.layers if l.last_attn_weights is not None]

