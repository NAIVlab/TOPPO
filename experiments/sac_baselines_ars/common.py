import collections
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import functools
import flax
import flax.linen as nn
from jax import lax
import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints
from jax.flatten_util import ravel_pytree

Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "rewards", "masks", "next_observations"]
)

default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros
EPS = 1e-8


def activation_func(activation_name: str = "relu"):
    if activation_name == "swiglu":
        activation_func = SwiGLU
    else:
        try:
            activation_func = getattr(nn, activation_name)
        except AttributeError:
            raise ValueError(
                f"Activation function '{activation_name}' is not found in flax.linen."
            )
    return activation_func


def save_model(model: "Model", checkpoint_dir: str, prefix: str = "checkpoint"):
    """
    Save the model state to the specified directory.

    Args:
        model (Model): The model instance to save.
            checkpoint_dir (str): Directory where checkpoints will be saved.
            prefix (str): Prefix for checkpoint filenames.

    Example:
        save_model(
            actor,
            checkpoint_dir="./models",
        )
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=model,
        step=model.step,
        prefix=prefix,
        overwrite=True,  # Set to False if you want to keep multiple checkpoints
    )


def load_model(
    checkpoint_dir: str,
    apply_fn: nn.Module,
    tx: Optional[optax.GradientTransformation] = None,
) -> "Model":
    """
    Load the model state from the specified directory.

    Args:
        checkpoint_dir (str): Directory from where to load checkpoints.
        apply_fn (nn.Module): The model architecture.
        tx (Optional[optax.GradientTransformation]): The optimizer used during training.

    Returns:
        Model: The loaded model instance.

    Example:
        pretrain_actor = load_model(
            apply_fn=actor_def,
            checkpoint_dir="./model",
            tx=optimiser,
        )
    """
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=None,  # We'll handle reconstruction manually
        step=None,  # Load the latest checkpoint
        prefix="checkpoint",
    )

    if restored is None:
        raise ValueError(f"No checkpoint found in directory {checkpoint_dir}")

    # Extract the saved state
    step = restored["step"]
    params = restored["params"]
    opt_state = restored["opt_state"]

    return Model(
        step=step, apply_fn=apply_fn, params=params, tx=tx, opt_state=opt_state
    )


def extract_dense_kernels(params):
    """
    Return a simple dict of base Dense kernels,
    e.g. {'dense_0': {'kernel': ...}, 'dense_1': {'kernel': ...}, ...}
    """
    base_dict = {}
    idx = 0
    for k, v in params.items():
        # Typically "Dense_0", "Dense_1", etc.
        if "Dense_" in k:
            base_dict[f"dense_{idx}"] = {"kernel": v["kernel"], "bias": v["bias"]}
            idx += 1
        elif k == "means" or k == "log_stds":
            base_dict[k] = {"kernel": v["kernel"], "bias": v["bias"]}
    return base_dict


def extract_ln_params(params):
    """
    Returns a dictionary that maps layernorm module names to (scale, bias).
    E.g. {
        'LayerNorm_0': { 'scale': ..., 'bias': ... },
        'LayerNorm_1': { 'scale': ..., 'bias': ... },
        ...
    }
    """
    ln_dict = {}
    for k, v in params.items():
        # If it looks like "LayerNorm_..." or your naming pattern
        if "LayerNorm_" in k:
            ln_dict[k] = {"scale": v["scale"], "bias": v["bias"]}
    return ln_dict


def flatten_taskwise_gradients(grads_pytree, n_task):
    """
    grads_pytree: a PyTree where each leaf has shape (n_task, ...)
    """
    # We'll 'ravel' just once on an example slice to get unravel_fn
    example_grad = jax.tree_util.tree_map(lambda x: x[0], grads_pytree)
    flat_example_grad, unravel_fn = ravel_pytree(example_grad)

    def slice_and_flatten(i):
        g_i = jax.tree_util.tree_map(lambda x: x[i], grads_pytree)
        g_i_flat, _ = ravel_pytree(g_i)
        return g_i_flat

    # vmap over 0..n_task
    grads_flat = jax.vmap(slice_and_flatten)(jnp.arange(n_task))
    return grads_flat, unravel_fn


@functools.partial(jax.jit)
def project_single_grad(grad_i, all_grads):
    """
    The typical PCGrad projection:
      for each grad g_k in all_grads, if dot(grad_i, g_k) < 0,
         subtract the projection of grad_i onto g_k from grad_i
    """

    @functools.partial(jax.jit)
    def body_fun(carry, g_k):
        dot_ik = jnp.dot(carry, g_k)
        proj_coeff = dot_ik / (jnp.dot(g_k, g_k) + 1e-9)
        # Only subtract if dot < 0
        carry = carry - jnp.minimum(proj_coeff, 0.0) * g_k
        return carry, proj_coeff

    final_grad, proj_coeffs = jax.lax.scan(body_fun, grad_i, all_grads)
    # (Optional) track how many negative dot products we encountered
    # or just skip it if you don't need that stat
    return final_grad, (proj_coeffs < 0).sum()  # or other info if desired


def gram_schmidt_single(sample_vectors):
    # no jit here
    n_models, dim = sample_vectors.shape
    basis = jnp.zeros((n_models, dim))
    v = sample_vectors[0]
    v = v / jnp.linalg.norm(v)
    basis = basis.at[0].set(v)
    # for i in range(1, n_models):
    #     v = sample_vectors[i]
    #     for j in range(i):
    #         v = v - jnp.dot(v, basis[j]) * basis[j]
    #     v = v / jnp.linalg.norm(v)
    #     basis = basis.at[i].set(v)

    for i in range(1, n_models):
        v = sample_vectors[i]
        B = basis[:i]
        coeffs = B @ v
        v = v - (coeffs @ B)
        v = v / (jnp.linalg.norm(v))
        basis = basis.at[i].set(v)
    return basis


def gram_schmidt_single2(x: jnp.ndarray) -> jnp.ndarray:
    n, d = x.shape
    basis0 = jnp.zeros((n, d), dtype=x.dtype)

    v0 = x[0]
    v0 = v0 / (jnp.linalg.norm(v0) + EPS)
    basis0 = basis0.at[0].set(v0)

    def body(i, basis):
        # project x[i] onto span(basis[:i])
        v = x[i]
        # Use full basis with zeros for rows >= i to keep shapes static.
        coeffs = basis @ v
        proj = coeffs @ basis

        w = v - proj
        w = w / (jnp.linalg.norm(w) + EPS)
        return basis.at[i].set(w)

    basis = lax.fori_loop(1, n, body, basis0)
    return basis


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.xavier_normal()


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any
InfoDict = Dict[str, float]


def SwiGLU(x):
    x, gate = jnp.split(x, 2, axis=-1)
    return nn.silu(gate) * x


class LoRAMultiDense(nn.Module):
    features: int
    rank: int = 4
    alpha: float = 1.0
    n_task: int = 1
    base_kernel: Optional[jnp.ndarray] = None
    base_bias: Optional[jnp.ndarray] = None
    """
    base_kernel will be the frozen weight (pretrained).
    We'll learn only the low-rank adapters for each task T_i.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Compute the base output from the frozen pretrained kernel if provided.
        if self.base_kernel is not None:
            base_output = jnp.einsum("tbi,if->tbf", x, self.base_kernel)
            if self.base_bias is not None:
                base_output += self.base_bias
        else:
            base_output = nn.Dense(self.features, use_bias=True)(x)

        # Initialize LoRA adapter parameters per task.
        lora_down = self.param(
            "lora_down",
            nn.initializers.lecun_normal(),
            (x.shape[-1], self.rank, self.n_task),
        )
        lora_up = self.param(
            "lora_up",
            nn.initializers.zeros,
            (self.rank, self.features, self.n_task),
        )

        # Compute the low-rank update using einsum:
        # For each task t, delta[t, b, f] = sum_{i,r} x[t, b, i] * lora_down[i, r, t] * lora_up[r, f, t]
        delta = jnp.einsum("tbi,irt,rft->tbf", x, lora_down, lora_up)
        scaling = self.alpha / self.rank
        delta = delta * scaling

        return base_output + delta


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = -1
    layer_norm: bool = False
    init_layer_norm: bool = False
    dtype: Optional[Dtype] = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if self.init_layer_norm:
            x = nn.LayerNorm()(x)
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, dtype=self.dtype, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
                if self.dropout_rate >= 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
        return x


class LoRAMulti_MLP(nn.Module):
    hidden_dims: Sequence[int]
    rank: int = 4
    alpha: float = 1.0
    n_task: int = 1
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    base_params: Optional[Dict] = None
    init_layer_norm: bool = False
    layer_norm: bool = False
    dropout_rate: float = -1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if x.ndim == 2:
            x = jnp.expand_dims(x, axis=1)
        # optional LN at input
        if self.init_layer_norm:
            x = nn.LayerNorm()(x)

        for i, size in enumerate(self.hidden_dims):
            # LoRA Dense
            base_kernel = None
            if self.base_params is not None:
                base_kernel = self.base_params[f"dense_{i}"]["kernel"]
                base_bias = self.base_params[f"dense_{i}"]["bias"]
            x = LoRAMultiDense(
                features=size,
                rank=self.rank,
                alpha=self.alpha,
                n_task=self.n_task,
                base_kernel=base_kernel,
                base_bias=base_bias,
                name=f"lora_dense_{i}",
            )(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
                # Optional dropout
                if self.dropout_rate >= 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
        return x


class ModularGatedNet(nn.Module):
    output_shape: int
    base_hidden_shapes: Sequence[int]
    em_hidden_shapes: Sequence[int]

    # The repeated layers / modules config:
    num_layers: int
    num_modules: int
    module_hidden: int

    # Gating config:
    gating_hidden: int
    num_gating_layers: int

    # Activations and inits:
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        embedding_input: jnp.ndarray,
        training: bool = False,
        return_weights: bool = False,
    ):
        # 1) base & em_base
        base_out = MLP(
            hidden_dims=self.base_hidden_shapes,
            activations=self.activations,
            activate_final=False,
        )(x, training=training)

        embedding = MLP(
            hidden_dims=self.em_hidden_shapes,
            activations=self.activations,
            activate_final=False,
        )(embedding_input, training=training)

        embedding = embedding * base_out

        out = self.activations(base_out)

        # 2) Gating MLP ("gating_fcs"): replicate your gating_fcs
        g = MLP(
            hidden_dims=[self.gating_hidden for _ in range(self.num_gating_layers)],
            activations=self.activations,
            activate_final=False,
        )(self.activations(embedding), training=training)

        # 3) gating_weight_fc_0 -> produce raw_weight for first layer with shape = (..., num_modules, num_modules)
        raw_weight = nn.Dense(
            features=self.num_modules * self.num_modules,
            kernel_init=self.kernel_init,
            name=f"gating_weight_cond_fc_0",
        )(self.activations(g))

        weights_list = []
        flatten_weights_list = []

        # shape = (..., num_modules, num_modules)
        shape_2d = raw_weight.shape[:-1] + (self.num_modules, self.num_modules)
        raw_weight = raw_weight.reshape(shape_2d)
        softmax_weight = nn.softmax(raw_weight, axis=-1)

        weights_list.append(softmax_weight)
        flatten_weights_list.append(
            softmax_weight.reshape(softmax_weight.shape[:-2] + (-1,))
        )

        # 4) Additional gating layers for layers (2,...,):
        for layer_idx in range(self.num_layers - 2):
            cond_fc = nn.Dense(
                features=self.gating_hidden,
                kernel_init=self.kernel_init,
                name=f"gating_weight_cond_fc_{layer_idx+1}",
            )
            # gating_weight_fc_(layer_idx+1):
            weight_fc = nn.Dense(
                features=self.num_modules * self.num_modules,
                kernel_init=self.kernel_init,
                name=f"gating_weight_fc_{layer_idx+1}",
            )

            concat_flat = jnp.concatenate(flatten_weights_list, axis=-1)

            cond_val = cond_fc(concat_flat)
            cond_val = cond_val * g
            cond_val = self.activations(cond_val)

            raw_w = weight_fc(cond_val)
            raw_w = raw_w.reshape(shape_2d)
            w = nn.softmax(raw_w, axis=-1)

            weights_list.append(w)
            flatten_weights_list.append(w.reshape(w.shape[:-2] + (-1,)))

        # 5) gating_weight_cond_last -> gating_weight_last => num_modules
        gw_cond_last = nn.Dense(
            features=self.gating_hidden,
            kernel_init=self.kernel_init,
            name="gating_weight_cond_last",
        )
        gw_last = nn.Dense(
            features=self.num_modules,
            kernel_init=self.kernel_init,
            name="gating_weight_last",
        )

        concat_flat = jnp.concatenate(flatten_weights_list, axis=-1)

        cond_val = gw_cond_last(concat_flat)
        cond_val = cond_val * g
        cond_val = self.activations(cond_val)

        raw_last_weight = gw_last(cond_val)  # shape (..., num_modules)
        last_weight = nn.softmax(raw_last_weight, axis=-1)

        # 6) Build the actual "layer modules"
        module_outputs = []
        for j in range(self.num_modules):
            out_j = nn.Dense(
                features=self.module_hidden,
                kernel_init=self.kernel_init,
                name=f"module_0_{j}",
            )(out)
            module_outputs.append(out_j[..., None, :])  # unsqueeze dim -2

        module_outputs = jnp.concatenate(module_outputs, axis=-2)

        for i in range(1, self.num_layers):
            new_module_outputs = []
            w_i = weights_list[i - 1]
            for j in range(self.num_modules):
                # Weighted sum across the "module_outputs"
                w_j = w_i[..., j, :][..., None]  # (..., num_modules, 1)
                module_input_j = (module_outputs * w_j).sum(axis=-2)

                module_input_j = self.activations(module_input_j)
                out_ij = nn.Dense(
                    features=self.module_hidden,
                    kernel_init=self.kernel_init,
                    name=f"module_{i}_{j}",
                )(module_input_j)
                new_module_outputs.append(out_ij[..., None, :])
            module_outputs = jnp.concatenate(new_module_outputs, axis=-2)

        out = (module_outputs * last_weight[..., None]).sum(axis=-2)
        out = self.activations(out)

        # Final linear
        final_dense = nn.Dense(features=self.output_shape, kernel_init=self.kernel_init)
        out = final_dense(out)

        if return_weights:
            return out, weights_list, last_weight
        return out


class OrthogonalLayer1D(nn.Module):
    """
    Gram-Schmidt across axis=0 (n_experts), for input shape:
        [n_experts, n_task, batch_size, dim].
    Outputs the same shape, with row vectors orthonormal per sample.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x shape => [n_experts, n_task, batch_size, dim]
        """

        # @jax.jit
        def forward_fn(x_jit: jnp.ndarray) -> jnp.ndarray:
            # 1) [n_experts, n_task, batch_size, dim] => [n_task, batch_size, n_experts, dim]
            x_t = x_jit.transpose(1, 2, 0, 3)

            # 2) Flatten [n_task, batch_size, n_experts, dim] => [N, n_experts, dim]
            n_task, batch_size, n_experts, dim = x_t.shape
            x_flat = x_t.reshape((n_task * batch_size, n_experts, dim))

            # 3) Gram-Schmidt over the flattened batch => [N, n_experts, dim]
            # batched_gram_schmidt = jax.jit(
            #     jax.vmap(gram_schmidt_single, in_axes=0, out_axes=0)
            # )
            batched_gram_schmidt = jax.vmap(gram_schmidt_single, in_axes=0, out_axes=0)
            # batched_gram_schmidt = jax.vmap(gram_schmidt_single2, in_axes=0, out_axes=0)

            orth_flat = batched_gram_schmidt(x_flat)
            # 4) Reshape back => [n_task, batch_size, n_experts, dim]
            orth_unflat = orth_flat.reshape((n_task, batch_size, n_experts, dim))
            return orth_unflat

        return forward_fn(x)


@flax.struct.dataclass
class Model:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        inputs: Sequence[jnp.ndarray],
        tx: Optional[optax.GradientTransformation] = None,
    ) -> "Model":
        variables = model_def.init(*inputs)
        params = variables.pop("params")
        opt_state = tx.init(params) if tx else None
        return cls(
            step=1,
            apply_fn=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
        )

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({"params": self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, "Model"]:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)

        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return (
            self.replace(
                step=self.step + 1, params=new_params, opt_state=new_opt_state
            ),
            info,
        )

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> "Model":
        with open(load_path, "rb") as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)


@flax.struct.dataclass
class MultiModel:
    step: int
    n_task: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        inputs: Sequence[jnp.ndarray],
        tx: Optional[optax.GradientTransformation] = None,
        n_task: int = 1,
    ) -> "Model":
        variables = model_def.init(*inputs)
        params = variables.pop("params")
        opt_state = tx.init(params) if tx else None
        return cls(
            step=1,
            apply_fn=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            n_task=n_task,
        )

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({"params": self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, "Model"]:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)

        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return (
            self.replace(
                step=self.step + 1, params=new_params, opt_state=new_opt_state
            ),
            info,
        )

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> "Model":
        with open(load_path, "rb") as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)


@flax.struct.dataclass
class MultiModelFairGrad:
    """MultiModel whose apply_gradient combines per-task gradients with
    FairGrad (Ban & Ji 2024, alpha=1 = Nash bargaining) instead of PCGrad
    projection — the critic-side balancing (FG-c) of the TOPPO critic stack.

    ``loss_fn`` must return a per-task loss VECTOR (same contract as the
    PCGrad loss fns); the (K, K) Gram system is solved on the host in fp64
    via jax.pure_callback (see fairgrad_host.py).  The combined gradient is
    ``sum_i w_i g_i`` with norm pinned at sqrt(K) (Theorem 1) — no rescale,
    matching the paper's fairgrad_rescale=False default.
    """

    step: int
    # static (not traced): n_task shapes jnp.arange in the taskwise flatten;
    # fairgrad_alpha is closed over by the pure_callback callable.
    n_task: int = flax.struct.field(pytree_node=False)
    fairgrad_alpha: float = flax.struct.field(pytree_node=False)
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        inputs: Sequence[jnp.ndarray],
        tx: Optional[optax.GradientTransformation] = None,
        n_task: int = 1,
        fairgrad_alpha: float = 1.0,
    ) -> "MultiModelFairGrad":
        variables = model_def.init(*inputs)
        params = variables.pop("params")
        opt_state = tx.init(params) if tx else None
        return cls(
            step=1,
            apply_fn=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            n_task=n_task,
            fairgrad_alpha=fairgrad_alpha,
        )

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({"params": self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn, n_task: int = 0):
        from fairgrad_host import fairgrad_combine_flat

        jacobian_fn = jax.jacrev(loss_fn, has_aux=True)
        grads_per_task, info = jacobian_fn(self.params)

        grads_per_task_flat, unravel_fn = flatten_taskwise_gradients(
            grads_per_task, n_task or self.n_task
        )  # (n_task, dim)

        combined_flat, weights = fairgrad_combine_flat(
            grads_per_task_flat, self.fairgrad_alpha
        )
        final_grad = unravel_fn(combined_flat)

        updates, new_opt_state = self.tx.update(final_grad, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        info = dict(info)
        info["fairgrad_w"] = weights

        return (
            self.replace(
                step=self.step + 1, params=new_params, opt_state=new_opt_state
            ),
            info,
        )

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> "MultiModelFairGrad":
        with open(load_path, "rb") as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)


@flax.struct.dataclass
class MultiModelPCGrad:
    step: int
    # static so apply_gradient can default to the real task count (upstream
    # hardcoded n_task=50 at the call sites via the parameter default, which
    # NaN-poisons params on any n_task != 50 — jnp.take fills OOB with NaN).
    # Values are identical on MT50, so MT50 behavior is unchanged.
    n_task: int = flax.struct.field(pytree_node=False)
    rng: jax.random.PRNGKey
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        inputs: Sequence[jnp.ndarray],
        tx: Optional[optax.GradientTransformation] = None,
        n_task: int = 1,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> "Model":
        variables = model_def.init(*inputs)
        params = variables.pop("params")
        opt_state = tx.init(params) if tx else None
        return cls(
            step=1,
            apply_fn=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            n_task=n_task,
            rng=rng,
        )

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({"params": self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn, n_task: int = 0) -> Tuple[Any, "Model"]:
        n_task = n_task or self.n_task
        jacobian_fn = jax.jacrev(loss_fn, has_aux=True)
        grads_per_task, info = jacobian_fn(self.params)
        # grads = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), jacobian)

        # ---------- PCGrad Projection Step -----------
        key, new_rng = jax.random.split(self.rng)
        shuffle_perm = jax.random.permutation(key, n_task)
        grads_per_task = jax.tree_util.tree_map(
            lambda x: jnp.take(x, shuffle_perm, axis=0), grads_per_task
        )

        # Flatten all tasks’ grads once
        grads_per_task_flat, unravel_fn = flatten_taskwise_gradients(
            grads_per_task, n_task
        )  # (n_task, dim)

        # Project each grad_i in grads_per_task_flat against the others
        proj_grads_flat, n_projections = jax.vmap(
            project_single_grad, in_axes=(0, None)
        )(grads_per_task_flat, grads_per_task_flat)

        # -----------------------------------------------------
        # Combine the projected gradients, e.g. average across tasks
        final_flat_grad = jnp.mean(proj_grads_flat, axis=0)
        final_grad = unravel_fn(final_flat_grad)

        updates, new_opt_state = self.tx.update(final_grad, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return (
            self.replace(
                step=self.step + 1,
                params=new_params,
                opt_state=new_opt_state,
                rng=new_rng,
            ),
            info,
        )
