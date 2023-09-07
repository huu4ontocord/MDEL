import warnings
from pathlib import Path
from typing import List, Tuple, Union

import fire
from torch import nn

from transformers import AutoTokenizer, PreTrainedModel, AutoModelForCausalLM
from transformers.utils import logging

logger = logging.get_logger(__name__)


def copy_layers(src_layers: nn.ModuleList, dest_layers: nn.ModuleList, layers_to_copy: List[int]) -> None:
    layers_to_copy = nn.ModuleList([src_layers[i] for i in layers_to_copy])
    assert len(dest_layers) == len(layers_to_copy), f"{len(dest_layers)} != {len(layers_to_copy)}"
    dest_layers.load_state_dict(layers_to_copy.state_dict())


LAYERS_TO_COPY = {
    # maps  num layers in teacher -> num_layers in student -> which teacher layers to copy.
    32: {
        1: [0],
        2: [0, 31],
        3: [0, 16, 31],
        # 4: [0, 10, 20, 31],
        # 6: [0, 3, 6, 9, 12, 31],
        8: [0, 4, 8, 12, 16, 20, 24, 31],
        # 9: [0, 1, 3, 5, 7, 9, 11, 13, 31],
        12: [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 31],
        16: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 31],
        32: list(range(32))
    },
}

LAYERS_TO_SUPERVISE = {
    # maps  num layers in student -> which teacher layers to copy.
    6: {1: [5], 2: [3, 5], 3: [1, 4, 5], 4: [1, 2, 4, 5]},
    12: {1: [11], 2: [5, 11], 3: [3, 7, 11], 6: [1, 3, 5, 8, 10, 11]},
    16: {1: [15], 4: [4, 9, 12, 15], 8: [1, 3, 5, 7, 9, 11, 13, 15]},
    32: {8: [3, 7, 11, 15, 19, 23, 27, 31], 16: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]},
}

def get_layers_to_supervise(n_student, n_teacher) -> List[int]:
    """Used or the --supervise_forward kwarg"""
    if n_student > n_teacher:
        raise ValueError(f"Cannot perform intermediate supervision for student {n_student} > teacher {n_teacher}")
    elif n_teacher == n_student:
        return list(range(n_teacher))
    elif n_student == 1:
        return [n_teacher - 1]
    else:
        return LAYERS_TO_SUPERVISE[n_teacher][n_student]


def pick_layers_to_copy(n_student, n_teacher):
    try:
        val = LAYERS_TO_COPY[n_teacher][n_student]
        return val
    except KeyError:
        if n_student != n_teacher:
            warnings.warn(
                f"no hardcoded layers to copy for teacher {n_teacher} -> student {n_student}, defaulting to first"
                f" {n_student}"
            )
        return list(range(n_student))


def create_student_by_copying_alternating_layers(
    teacher: Union[str, PreTrainedModel],
    save_path: Union[str, Path] = "student",
    d: Union[int, None] = None,
    copy_first_teacher_layers=False,
    d_layers_to_copy=None,
    **extra_config_kwargs,
) -> Tuple[PreTrainedModel, List[int], List[int]]:
    """Make a student by copying alternating layers from a teacher, save it to save_path.
    Args:
        teacher: str or PreTrainedModel if str, this will call AutoModelForCausalLM.from_pretrained(teacher) before
        copying layers
        save_path: where to save the student, defaults to student directory.
        d: how many Decoder layers should the student have, default is fully copy of teacher
        copy_first_teacher_layers: [bool] dont copy alternating layers, just the first e/d.
        **extra_config_kwargs: extra kwargs to pass to the student, by default the teacher config is used.

    Returns:
        student: new, smaller model.  (Also saves it to save_path)
        d_layers_to_copy: list of which teacher decoder layers were used
    """
    _msg = "decoder_layers cannot be both None-- you would just have an identical teacher."
    assert (d is not None), _msg
    if isinstance(teacher, str):
        AutoTokenizer.from_pretrained(teacher).save_pretrained(save_path)  # purely for convenience
        teacher = AutoModelForCausalLM.from_pretrained(teacher).eval()
    else:
        assert isinstance(teacher, PreTrainedModel), f"teacher must be a model or string got type {type(teacher)}"
    init_kwargs = teacher.config.to_diff_dict()

    try:
        teacher_d = teacher.config.decoder_layers
        if d is None:
            d = teacher_d
        init_kwargs.update({"n_layer": d})
    except AttributeError:  # T5
        if hasattr(teacher.config, "n_layer"):
            teacher_d = teacher.config.n_layer
        else:
            teacher_d = teacher.config.n_layer
        if d is None:
            d = teacher_d
        if hasattr(teacher.config, "n_layer"):
            init_kwargs.update({"n_layer": d})
        else:
            init_kwargs.update({"n_layer": d})

    # Kwargs to instantiate student: teacher kwargs with updated layer numbers + **extra_config_kwargs
    init_kwargs.update(extra_config_kwargs)

    # Copy weights
    student_cfg = teacher.config_class(**init_kwargs)
    student = AutoModelForCausalLM.from_config(student_cfg, trust_remote_code=True)
    # Start by copying the full teacher state dict this will copy the first N teacher layers to the student.
    info = student.load_state_dict(teacher.state_dict(), strict=False)
    assert info.missing_keys == [], info.missing_keys  # every student key should have a teacher keys.

    if copy_first_teacher_layers:  # Our copying is done. We just log and save
        d_layers_to_copy = list(range(d))
        logger.info(
            f"Copied decoder layers {d_layers_to_copy}. Saving them to"
            f" {save_path}"
        )
        student.save_pretrained(save_path)
        return student, d_layers_to_copy

    # Decide which layers of the teacher to copy. Not exactly alternating -- we try to keep first and last layer.
    if d_layers_to_copy is None:
        d_layers_to_copy: List[int] = pick_layers_to_copy(d, teacher_d)

    try:
        if hasattr(
            teacher, "prophetnet"
        ): 
            copy_layers(teacher.prophetnet.decoder.layers, student.prophetnet.decoder.layers, d_layers_to_copy)
        else:
            copy_layers(teacher.model.decoder.layers, student.model.decoder.layers, d_layers_to_copy)
    except AttributeError:  
        copy_layers(teacher.transformer.h, student.transformer.h, d_layers_to_copy)
    logger.info(
        f"Copied decoder layers {d_layers_to_copy}. Saving them to {save_path}"
    )
    student.config.init_metadata = {
        "teacher_type": teacher.config.model_type,
        "copied_decoder_layers": d_layers_to_copy,
    }
    student.save_pretrained(save_path)
    # Save information about copying for easier reproducibility

    return student, d_layers_to_copy


if __name__ == "__main__":
    fire.Fire(create_student_by_copying_alternating_layers)