# #Copyright @2023 Xianhang Li
#
# # This code is based on materials from the Big Vision [https://github.com/google-research/big_vision].
# # Thanks to Big Vision  for their contributions to the field of computer vision and for their open-source contributions to this project.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer encoders both for text and for images."""

import importlib
from typing import Any, Optional, Tuple, Union

from helpers import utils
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

ConfigDict = Any


class Model(nn.Module):
    """Two towers transformer."""
    image: Optional[ConfigDict] = None
    text: Optional[ConfigDict] = None
    text_model: str = "proj.image_text.text_transformer"
    image_model: str = "vit"
    out_dim: Union[int, Tuple[int, int]] = 128
    temperature_init: float = 1.0

    @nn.compact
    def __call__(
            self,
            image,
            text=None,
            mask_ratio=0.0,
            mae_layer=-1,
            rng_mask=None,
            train=False,
            **kw):
        """Returns (B,C) image and (B,C) text representations."""

        # Support calling without text or without image, for example for
        # few-shot.
        ztxt, zimg = None, None
        out = {}
        out_dims = self.out_dim
        if isinstance(out_dims, int):
            out_dims = (out_dims, out_dims)

        # Embed the text:
        if text is not None:
            text_model = importlib.import_module(
                f"models.{self.text_model}"
            ).Model(**{"num_classes": out_dims[1], **(self.text or {})}, name="txt")

        if text is not None:
            ztxt, out_txt = text_model(text, **kw)
            for k, v in out_txt.items():
                out[f"txt/{k}"] = v

            # Normalize the embeddings the models give us.
            out["txt/norm"] = jnp.linalg.norm(ztxt, axis=1, keepdims=True)
            out["txt/normalized"] = ztxt = ztxt / (out["txt/norm"] + 1e-8)
            
        if image is not None:
            image_model = importlib.import_module(f"models.{self.image_model}").Model(
                **{"num_classes": out_dims[0], **(self.image or {})}, name="img", **kw)  # pylint: disable=not-a-mapping
            if mae_layer != -1:
                zimg, out_img = image_model(
                    image, mask_ratio=mask_ratio, mae_layer=mae_layer, rng_mask=rng_mask, **kw)
            else:
                zimg, out_img = image_model(
                    image, mask_ratio=mask_ratio, train=train, **kw)
            for k, v in out_img.items():
                out[f"img/{k}"] = v

            # Normalize the embeddings the models give us.
            out["img/norm"] = jnp.linalg.norm(zimg, axis=1, keepdims=True)
            out["img/normalized"] = zimg = zimg / (out["img/norm"] + 1e-8)

        temp_init = jnp.log(self.temperature_init)
        t = self.param("t", lambda key, shape, dtype: temp_init *
                       jnp.ones(shape, dtype), (1,), jnp.float32)
        out["t"] = jnp.exp(t)
        out["t/parameter"] = t

        return zimg, ztxt, out


def load(init_params, init_files, model_cfg, img_load_kw={}, txt_load_kw={}):  # pylint: disable=dangerous-default-value
    """Loads both towers, `init_files` is now a dict with `img` and `txt` keys."""
    if isinstance(init_files, str):
        # A shortcut for a single file checkpoint of a two_towers model.
        init_files = {k: f"{init_files}:{k}" for k in ("img", "txt", "t")}
    else:
        # Shallow copy because we'll pop stuff off.
        init_files = {**init_files}

    restored_params = {**init_params}

    img_init = init_files.pop("image", init_files.pop("img", None))
    if img_init:
        restored_params["img"] = importlib.import_module(
            f"models.{model_cfg.image_model}"
        ).load(init_params["img"], img_init, model_cfg.image, **img_load_kw)

    txt_init = init_files.pop("text", init_files.pop("txt", None))
    if txt_init:
        restored_params["txt"] = importlib.import_module(
            f"models.{model_cfg.text_model}"
        ).load(init_params["txt"], txt_init, model_cfg.text, **txt_load_kw)

    t_init = init_files.pop("temperature", init_files.pop("t", None))
    if t_init:
        restored_params["t"] = utils.load_params(None, t_init)

    assert not init_files, (
        f"There's something unused left in `config.model_init`. You probably got "
        f"a typo. Here it is: {init_files}")

    return restored_params
