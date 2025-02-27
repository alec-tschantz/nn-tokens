import jax
import jax.numpy as jnp
import equinox as eqx


class Tokenizer(eqx.Module):
    codes: jnp.ndarray
    num_codes: int
    distance_threshold: float
    max_codes: int = eqx.field(static=True)
    no_code_id: int = eqx.field(static=True)


def _squared_cdist(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    a2 = jnp.sum(a**2, axis=-1, keepdims=True)
    b2 = jnp.sum(b**2, axis=-1, keepdims=True)
    ab = a @ b.T
    return jnp.clip(a2 - 2 * ab + b2.T, a_min=0, a_max=None)


def forward_tokenizer(
    tokenizer: Tokenizer, x: jnp.ndarray  # shape [batch, dim]
) -> jnp.ndarray:
    # If tokenizer is empty, return no_code_id for all
    if tokenizer.num_codes == 0:
        return jnp.full((x.shape[0],), tokenizer.no_code_id, dtype=jnp.int32)

    # Compute pairwise distances to all codes [batch, max_codes]
    dist_sq = _squared_cdist(x, tokenizer.codes)

    # Create mask for valid indices
    valid_mask = jnp.arange(tokenizer.max_codes) < tokenizer.num_codes
    dist_sq_masked = jnp.where(valid_mask[None, :], dist_sq, jnp.inf)

    # Nearest neighbor index
    nearest_ids = jnp.argmin(dist_sq_masked, axis=-1)
    return nearest_ids


def train_tokenizer(
    tokenizer: Tokenizer, x: jnp.ndarray  # shape [batch, dim]
) -> tuple[Tokenizer, jnp.ndarray]:
    # If the codebook is empty and x is not, seed with the first vector.
    tokenizer = jax.lax.cond(
        (tokenizer.num_codes == 0) & (x.shape[0] > 0),
        lambda tok: eqx.tree_at(
            lambda t: (t.codes, t.num_codes),
            tok,
            (tok.codes.at[0].set(x[0]), 1),  # add the first sample at index 0
        ),
        lambda tok: tok,
        tokenizer,
    )

    # Distances to all codes [batch, max_codes]
    dist_sq = _squared_cdist(x, tokenizer.codes)
    valid_mask = jnp.arange(tokenizer.max_codes) < tokenizer.num_codes
    dist_sq_masked = jnp.where(valid_mask[None, :], dist_sq, jnp.inf)

    # Get nearest neighbor + distance
    nearest_ids = jnp.argmin(dist_sq_masked, axis=-1)
    min_dist_sq = jnp.min(dist_sq_masked, axis=-1)  # shape [batch]

    # Identify which samples are out of threshold
    outside = min_dist_sq > tokenizer.distance_threshold
    any_outside = jnp.any(outside)

    def no_new_codes_fn(tok_and_ids):
        return tok_and_ids

    def add_codes_fn(tok_and_ids):
        tok, old_ids = tok_and_ids

        def body_fn(tok_inner, idx):
            cond = outside[idx] & (tok_inner.num_codes < tok_inner.max_codes)

            def do_add_code():
                new_idx = tok_inner.num_codes
                code_to_add = x[idx]
                new_codes = tok_inner.codes.at[new_idx].set(code_to_add)
                return eqx.tree_at(
                    lambda t: (t.codes, t.num_codes),
                    tok_inner,
                    (new_codes, new_idx + 1),
                )

            return jax.lax.cond(cond, do_add_code, lambda: tok_inner), None

        tok_out, _ = jax.lax.scan(body_fn, tok, jnp.arange(x.shape[0]))

        dist_sq2 = _squared_cdist(x, tok_out.codes)
        valid_mask2 = jnp.arange(tok_out.max_codes) < tok_out.num_codes
        dist_sq2_masked = jnp.where(valid_mask2[None, :], dist_sq2, jnp.inf)
        new_ids = jnp.argmin(dist_sq2_masked, axis=-1)
        return (tok_out, new_ids)

    tok_and_ids = (tokenizer, nearest_ids)
    tokenizer, new_ids = jax.lax.cond(
        any_outside, add_codes_fn, no_new_codes_fn, tok_and_ids
    )
    return tokenizer, new_ids
