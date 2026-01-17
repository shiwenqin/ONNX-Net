import ast
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import onnx
from onnx import shape_inference
from transformers.tokenization_utils import PreTrainedTokenizer


def list_onnx_files(folder_path: Path) -> list[Path]:
    """Return a list of all .onnx file paths found within 'folder_path' (recursively)."""
    onnx_files = []
    for root, _, files in os.walk(str(folder_path)):
        for file in files:
            if file.lower().endswith(".onnx"):
                full_path = Path(root) / Path(file)
                onnx_files.append(full_path)
    return onnx_files


class ONNXConverter:
    """Class contains multiple methods to convert ONNX model to string representation.
    The one used in the paper is get_onnx_str(mode ='chain_slim').

    Variants in ablation study can be found with:
    - get_onnx_str(mode ='chain_slim_base')
    - get_onnx_str(mode ='chain_slim_param')
    - get_onnx_str(mode ='chain_slim_outshape')
    - get_onnx_str(mode ='chain_slim_input')
    """

    def __init__(self, onnx_path: Path, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        self.onnx_model = onnx.load(onnx_path)
        self.tokenizer = tokenizer

        self.onnx_model = onnx.shape_inference.infer_shapes(self.onnx_model)
        # self.onnx_model = onnx.version_converter.convert_version(self.onnx_model, 18)

    def _attrtype_to_str(self, x: onnx.AttributeProto) -> str:  # noqa: C901, PLR0911, PLR0912, PLR6301
        match x.type:
            case onnx.AttributeProto.FLOAT:
                return str(x.floats)
            case onnx.AttributeProto.FLOATS:
                return str(x.floats).replace(" ", "")
            case onnx.AttributeProto.INT:
                return str(x.ints)
            case onnx.AttributeProto.INTS:
                return str(x.ints).replace(" ", "")
            case onnx.AttributeProto.STRING:
                return str(x.strings)
            case onnx.AttributeProto.STRINGS:
                return str(x.strings).replace(" ", "")
            case onnx.AttributeProto.TENSOR:
                return "<BLOB>"
            case onnx.AttributeProto.TENSORS:
                return "<BLOB,...>"
            case onnx.AttributeProto.GRAPH:
                return "<BLOB>"
            case onnx.AttributeProto.GRAPHS:
                return "<BLOB,...>"
            case onnx.AttributeProto.SPARSE_TENSOR:
                return "<BLOB>"
            case onnx.AttributeProto.SPARSE_TENSORS:
                return "<BLOB,...>"
            case onnx.AttributeProto.TYPE_PROTO:
                return "<TYPE>"
            case onnx.AttributeProto.TYPE_PROTOS:
                return "<TYPE,...>"
            case onnx.AttributeProto.UNDEFINED:
                return "undefined"
            case _:
                msg = f"Unknown attribute type: {x.type}"
                raise ValueError(msg)

    def _get_token_count(self, model_str: str, tokenizer: PreTrainedTokenizer) -> int:  # noqa: PLR6301
        return len(tokenizer.tokenize(model_str, add_special_tokens=False))

    def get_onnx_infos(self) -> list[str]:  # noqa: C901
        # Generate new names for input-/output-/intermediate values
        def next_name() -> str:
            ret = f"Param{next_name.i}"  # pyright: ignore [reportFunctionMemberAccess]
            next_name.i += 1  # pyright: ignore [reportFunctionMemberAccess]
            return ret

        next_name.i = 1  # pyright: ignore [reportFunctionMemberAccess]
        name_map = defaultdict(next_name)
        for i, node in enumerate(self.onnx_model.graph.input, 1):
            name_map[node.name] = f"In{i}"
        for i, node in enumerate(self.onnx_model.graph.node, 1):
            for output in node.output:
                name_map[output] = f"Value{i}"
        for i, node in enumerate(self.onnx_model.graph.output, 1):
            name_map[node.name] = f"Out{i}"

        # Generate string representation of the model and node indices in the adjacency matrix
        node_idcs = {}
        str_rep = []
        for i, node in enumerate(self.onnx_model.graph.input):
            node_idcs[node.name] = i
            str_rep.append(
                f"{name_map[node.name]}:\nOp: Input({tuple(x.dim_value for x in node.type.tensor_type.shape.dim)})",
            )
        for i, node in enumerate(self.onnx_model.graph.node, max(node_idcs.values()) + 1):
            node_rep = ""
            node_rep += f"{','.join([name_map[x] for x in node.input]) if len([name_map[x] for x in node.input]) > 0 else 'Const'}-->{','.join([name_map[x] for x in node.output])}:\n"  # noqa: E501
            node_rep += f"Op: {node.op_type}({','.join([f'{x.name}={self._attrtype_to_str(x)}' for x in node.attribute if self._attrtype_to_str(x) != '[]'])})"  # noqa: E501
            params = [
                f"{name_map[x.name]}: {str(x.dims).replace(' ', '')}"
                for x in self.onnx_model.graph.initializer
                if x.name in node.input
            ]
            if len(params) > 0:
                # node_rep += f"\n{'\n'.join(params)}"
                node_rep += "\n" + "\n".join(params)

            str_rep.append(node_rep)
            node_idcs[node.name] = i
            for output in node.output:  # Make output of nodes refer back to the node
                node_idcs[output] = i
        for node in sorted(self.onnx_model.graph.output, key=lambda x: node_idcs[x.name]):
            str_rep.append(  # noqa: PERF401
                f"{name_map[node.name]}:\nOp: Output({tuple(x.dim_value for x in node.type.tensor_type.shape.dim)})",
            )

        return str_rep

    def get_onnx_infos_slim(self) -> list[str]:  # noqa: C901, PLR0912
        """Remove Input section, merge the input shape into Op section."""  # noqa: DOC201

        # Generate new names for input-/output-/intermediate values
        def next_name() -> str:
            ret = f"Param{next_name.i}"  # pyright: ignore [reportFunctionMemberAccess]
            next_name.i += 1  # pyright: ignore [reportFunctionMemberAccess]
            return ret

        next_name.i = 1  # pyright: ignore [reportFunctionMemberAccess]
        name_map = defaultdict(next_name)
        for i, node in enumerate(self.onnx_model.graph.input, 1):
            name_map[node.name] = f"In{i}"
        for i, node in enumerate(self.onnx_model.graph.node, 1):
            for output in node.output:
                name_map[output] = f"Value{i}"
        for i, node in enumerate(self.onnx_model.graph.output, 1):
            name_map[node.name] = f"Out{i}"

        # Generate string representation of the model and node indices in the adjacency matrix
        node_idcs = {}
        in_weight_map = {}
        str_rep = []
        for i, node in enumerate(self.onnx_model.graph.input):
            node_idcs[node.name] = i
            in_weight_map[node.name] = ",".join([str(x.dim_value) for x in node.type.tensor_type.shape.dim])
        for i, node in enumerate(self.onnx_model.graph.node, max(node_idcs.values()) + 1):
            node_rep = ""
            if len([name_map[x] for x in node.input]) > 0:
                input_str = ""
                for x in node.input:
                    if x in in_weight_map:
                        input_str += f"{name_map[x]}({in_weight_map[x]}),"
                    else:
                        input_str += f"{name_map[x]},"
                input_part = input_str[:-1]  # Remove trailing comma
            else:
                input_part = "Const"

            output_names = [name_map[x] for x in node.output]
            output_part = ",".join(output_names)

            node_rep += f"{input_part}-->{output_part}:\n"
            node_rep += f"Op: {node.op_type}({','.join([f'{x.name}={self._attrtype_to_str(x)}' for x in node.attribute if self._attrtype_to_str(x) != '[]'])})"  # noqa: E501
            params = [
                f"{name_map[x.name]}: {str(x.dims).replace(' ', '')}"
                for x in self.onnx_model.graph.initializer
                if x.name in node.input
            ]
            if len(params) > 0:
                # node_rep += f"\n{'\n'.join(params)}"
                node_rep += "\n" + "\n".join(params)

            str_rep.append(node_rep)
            node_idcs[node.name] = i
            for output in node.output:  # Make output of nodes refer back to the node
                node_idcs[output] = i
        for node in sorted(self.onnx_model.graph.output, key=lambda x: node_idcs[x.name]):
            str_rep.append(  # noqa: PERF401
                f"{name_map[node.name]}:\nOp: Output({tuple(x.dim_value for x in node.type.tensor_type.shape.dim)})",
            )

        return str_rep

    def get_onnx_infos_slim_noidentity(self) -> list[str]:  # noqa: C901, PLR0912
        # Generate new names for input-/output-/intermediate values
        def next_name() -> str:
            ret = f"Param{next_name.i}"  # pyright: ignore [reportFunctionMemberAccess]
            next_name.i += 1  # pyright: ignore [reportFunctionMemberAccess]
            return ret

        next_name.i = 1  # pyright: ignore [reportFunctionMemberAccess]
        name_map = defaultdict(next_name)
        for i, node in enumerate(self.onnx_model.graph.input, 1):
            name_map[node.name] = f"In{i}"
        for i, node in enumerate(self.onnx_model.graph.node, 1):
            for output in node.output:
                name_map[output] = f"Value{i}"
        for i, node in enumerate(self.onnx_model.graph.output, 1):
            name_map[node.name] = f"Out{i}"

        # Generate string representation of the model and node indices in the adjacency matrix
        node_idcs = {}
        in_weight_map = {}
        str_rep = []
        for i, node in enumerate(self.onnx_model.graph.input):
            node_idcs[node.name] = i
            in_weight_map[node.name] = ",".join([str(x.dim_value) for x in node.type.tensor_type.shape.dim])
        for i, node in enumerate(self.onnx_model.graph.node, max(node_idcs.values()) + 1):
            node_rep = ""
            if len([name_map[x] for x in node.input]) > 0:
                input_str = ""
                for x in node.input:
                    if x in in_weight_map:
                        input_str += f"{name_map[x]}({in_weight_map[x]}),"
                    else:
                        input_str += f"{name_map[x]},"
                input_part = input_str[:-1]  # Remove trailing comma
            else:
                input_part = "Const"

            if node.op_type == "Identity" and node.input[0] in in_weight_map:
                for x in node.output:
                    in_weight_map[x] = in_weight_map[node.input[0]]
            else:
                output_names = [name_map[x] for x in node.output]
                output_part = ",".join(output_names)

                node_rep += f"{input_part}-->{output_part}:\n"
                node_rep += f"Op: {node.op_type}({','.join([f'{x.name}={self._attrtype_to_str(x)}' for x in node.attribute if self._attrtype_to_str(x) != '[]'])})"  # noqa: E501
                params = [
                    f"{name_map[x.name]}: {str(x.dims).replace(' ', '')}"
                    for x in self.onnx_model.graph.initializer
                    if x.name in node.input
                ]
                if len(params) > 0:
                    # node_rep += f"\n{'\n'.join(params)}"
                    node_rep += "\n" + "\n".join(params)

                str_rep.append(node_rep)
                node_idcs[node.name] = i
                for output in node.output:  # Make output of nodes refer back to the node
                    node_idcs[output] = i
        for node in sorted(self.onnx_model.graph.output, key=lambda x: node_idcs[x.name]):
            str_rep.append(  # noqa: PERF401
                f"{name_map[node.name]}:\nOp: Output({tuple(x.dim_value for x in node.type.tensor_type.shape.dim)})",
            )

        return str_rep

    def get_onnx_infos_oponly(self) -> list[str]:
        node_idcs = {}
        str_rep = []
        for i, node in enumerate(self.onnx_model.graph.node):
            node_rep = ""
            node_rep += f"Op: {node.op_type}({','.join([f'{x.name}={self._attrtype_to_str(x)}' for x in node.attribute if self._attrtype_to_str(x) != '[]'])})"  # noqa: E501
            str_rep.append(node_rep)
            node_idcs[node.name] = i
            for output in node.output:  # Make output of nodes refer back to the node
                node_idcs[output] = i
        for node in sorted(self.onnx_model.graph.output, key=lambda x: node_idcs[x.name]):
            str_rep.append(f"Op: Output({tuple(x.dim_value for x in node.type.tensor_type.shape.dim)})")  # noqa: PERF401

        return str_rep

    def get_onnx_infos_oponly_extended(self) -> list[str]:
        node_idcs = {}
        str_rep = []
        for i, node in enumerate(self.onnx_model.graph.node):
            node_rep = ""
            node_rep += f"Op: {node.op_type}({','.join([f'{x.name}={self._attrtype_to_str(x)}' for x in node.attribute if self._attrtype_to_str(x) != '[]'])})"  # noqa: E501
            node_rep += f", Inputs: {len(node.input)}, Outputs: {len(node.output)}"
            str_rep.append(node_rep)
            node_idcs[node.name] = i
            for output in node.output:  # Make output of nodes refer back to the node
                node_idcs[output] = i
        for node in sorted(self.onnx_model.graph.output, key=lambda x: node_idcs[x.name]):
            str_rep.append(f"Op: Output({tuple(x.dim_value for x in node.type.tensor_type.shape.dim)})")  # noqa: PERF401

        return str_rep

    def get_onnx_infos_template_compressed(  # noqa: C901
        self,
    ) -> list[str]:
        # Generate new names for input-/output-/intermediate values
        def next_name() -> str:
            ret = f"Param{next_name.i}"  # pyright: ignore [reportFunctionMemberAccess]
            next_name.i += 1  # pyright: ignore [reportFunctionMemberAccess]
            return ret

        next_name.i = 1  # pyright: ignore [reportFunctionMemberAccess]
        name_map = defaultdict(next_name)
        for i, node in enumerate(self.onnx_model.graph.input, 1):
            name_map[node.name] = f"In{i}"
        for i, node in enumerate(self.onnx_model.graph.node, 1):
            for output in node.output:
                name_map[output] = f"Value{i}"
        for i, node in enumerate(self.onnx_model.graph.output, 1):
            name_map[node.name] = f"Out{i}"

        OP_DESCRIPTOR_FUNCS = {  # noqa: N806
            "Input": lambda _: "IN()",
            "Constant": lambda _: "CON()",
            "Output": lambda _: "OUT()",
            "Conv": lambda n: (
                "C("
                + str(next(filter(lambda x: x.name == "kernel_shape", n.attribute)).ints[0])
                + ","
                + str(next(filter(lambda x: x.name == "strides", n.attribute)).ints[0])
                + ","
                + str(next(filter(lambda x: x.name == "pads", n.attribute)).ints[0])
                + ")"
            ),
            "Relu": lambda _: "R()",
            "Add": lambda _: "A()",
            "MaxPool": lambda n: f"MP({next(filter(lambda x: x.name == 'kernel_shape', n.attribute)).ints[0]!s})",
            "Concat": lambda _: "CC()",
            "MatMul": lambda _: "MM()",
            "Gemm": lambda _: "MM()",
            "BatchNormalization": lambda _: "BN()",
            "AveragePool": lambda n: f"AP({next(filter(lambda x: x.name == 'kernel_shape', n.attribute)).ints[0]!s})",
            "Slice": lambda _: "S()",
        }

        # Generate string representation of the model and node indices in the adjacency matrix
        node_idcs = {}
        str_rep = [
            "The following text describes a neural network and uses this grammar:\n"
            + "<ENTRY> := <INPUTS>--><OUTPUTS><OPERATION>\n"
            + "<OPERATION> := [\n"
            + "    <INPUT> := IN\\(\\)\n"
            + "    <CONSTANT> := CON\\(\\)\n"
            + "    <OUTPUT> := OUT\\(\\)\n"
            + "    <CONV> := C\\(<KERNEL_SIZE>,<STRIDE>,<PADDING>\\)\n"
            + "    <RELU> := R\\(\\)\n"
            + "    <ADD> := A\\(\\)\n"
            + "    <MAXPOOL> := MP\\(<KERNEL_SIZE>\\)\n"
            + "    <CONCAT> := CC\\(\\)\n"
            + "    <MATMUL> := MM\\(\\)\n"
            + "    <BATCHNORM> := BN\\(\\)\n"
            + "    <AVGPOOL> := AP\\(<KERNEL_SIZE>\\)\n"
            + "    <SLICE> := S\\(\\)\n"
            + "]\n"
            + "Unkown operations are printed in a generic format with full names and attributes.",
        ]
        for i, node in enumerate(self.onnx_model.graph.input):
            node_idcs[node.name] = i
            str_rep.append(
                f"{name_map[node.name]}:\nOp: Input({tuple(x.dim_value for x in node.type.tensor_type.shape.dim)})",
            )
        for i, node in enumerate(
            self.onnx_model.graph.node,
            max(node_idcs.values()) + 1,
        ):
            node_rep = ""
            if node.op_type in OP_DESCRIPTOR_FUNCS:
                node_rep += f"{','.join([name_map[x] for x in node.input]) if len([name_map[x] for x in node.input]) > 0 else 'Const'}-->{','.join([name_map[x] for x in node.output])}:"  # noqa: E501
                node_rep += OP_DESCRIPTOR_FUNCS[node.op_type](node)
            else:
                node_rep += f"{','.join([name_map[x] for x in node.input]) if len([name_map[x] for x in node.input]) > 0 else 'Const'}-->{','.join([name_map[x] for x in node.output])}:\n"  # noqa: E501
                node_rep += f"Op: {node.op_type}({','.join([f'{x.name}={self._attrtype_to_str(x)}' for x in node.attribute if self._attrtype_to_str(x) != '[]'])})"  # noqa: E501
                params = [
                    f"{name_map[x.name]}: {str(x.dims).replace(' ', '')}"
                    for x in self.onnx_model.graph.initializer
                    if x.name in node.input
                ]
                if len(params) > 0:
                    # node_rep += f"\n{'\n'.join(params)}"
                    node_rep += "\n" + "\n".join(params)
            str_rep.append(node_rep)
            node_idcs[node.name] = i
            for output in node.output:  # Make output of nodes refer back to the node
                node_idcs[output] = i
        for node in sorted(self.onnx_model.graph.output, key=lambda x: node_idcs[x.name]):
            str_rep.append(  # noqa: PERF401
                f"{name_map[node.name]}:\nOp: Output({tuple(x.dim_value for x in node.type.tensor_type.shape.dim)})",
            )
        return str_rep

    def get_onnx_infos_template_compressed_slim(self) -> list[str]:  # noqa: C901, PLR0912

        # Generate new names for input-/output-/intermediate values
        def next_name() -> str:
            ret = f"Param{next_name.i}"  # pyright: ignore [reportFunctionMemberAccess]
            next_name.i += 1  # pyright: ignore [reportFunctionMemberAccess]
            return ret

        next_name.i = 1  # pyright: ignore [reportFunctionMemberAccess]
        name_map = defaultdict(next_name)
        for i, node in enumerate(self.onnx_model.graph.input, 1):
            name_map[node.name] = f"In{i}"
        for i, node in enumerate(self.onnx_model.graph.node, 1):
            for output in node.output:
                name_map[output] = f"Value{i}"
        for i, node in enumerate(self.onnx_model.graph.output, 1):
            name_map[node.name] = f"Out{i}"

        OP_DESCRIPTOR_FUNCS = {  # noqa: N806
            "Input": lambda _: "IN()",
            "Constant": lambda _: "CON()",
            "Output": lambda _: "OUT()",
            "Conv": lambda n: (
                "C("
                + str(next(filter(lambda x: x.name == "kernel_shape", n.attribute)).ints[0])
                + ","
                + str(next(filter(lambda x: x.name == "strides", n.attribute)).ints[0])
                + ","
                + str(next(filter(lambda x: x.name == "pads", n.attribute)).ints[0])
                + ")"
            ),
            "Relu": lambda _: "R()",
            "Add": lambda _: "A()",
            "MaxPool": lambda n: f"MP({next(filter(lambda x: x.name == 'kernel_shape', n.attribute)).ints[0]!s})",
            "Concat": lambda _: "CC()",
            "MatMul": lambda _: "MM()",
            "Gemm": lambda _: "MM()",
            "BatchNormalization": lambda _: "BN()",
            "AveragePool": lambda n: f"AP({next(filter(lambda x: x.name == 'kernel_shape', n.attribute)).ints[0]!s})",
            "Slice": lambda _: "S()",
        }

        # Generate string representation of the model and node indices in the adjacency matrix
        node_idcs = {}
        in_weight_map = {}
        str_rep = [
            "The following text describes a neural network and uses this grammar:\n"
            + "<ENTRY> := <INPUTS>--><OUTPUTS><OPERATION>\n"
            + "<OPERATION> := [\n"
            + "    <INPUT> := IN\\(<SHAPE>\\)\n"
            + "    <CONSTANT> := CON\\(\\)\n"
            + "    <OUTPUT> := OUT\\(\\)\n"
            + "    <CONV> := C\\(<KERNEL_SIZE>,<STRIDE>,<PADDING>\\)\n"
            + "    <RELU> := R\\(\\)\n"
            + "    <ADD> := A\\(\\)\n"
            + "    <MAXPOOL> := MP\\(<KERNEL_SIZE>\\)\n"
            + "    <CONCAT> := CC\\(\\)\n"
            + "    <MATMUL> := MM\\(\\)\n"
            + "    <BATCHNORM> := BN\\(\\)\n"
            + "    <AVGPOOL> := AP\\(<KERNEL_SIZE>\\)\n"
            + "    <SLICE> := S\\(\\)\n"
            + "]\n"
            + "Unkown operations are printed in a generic format with full names and attributes.",
        ]
        for i, node in enumerate(self.onnx_model.graph.input):
            node_idcs[node.name] = i
            in_weight_map[node.name] = ",".join([str(x.dim_value) for x in node.type.tensor_type.shape.dim])
        for i, node in enumerate(
            self.onnx_model.graph.node,
            max(node_idcs.values()) + 1,
        ):
            node_rep = ""
            if len([name_map[x] for x in node.input]) > 0:
                input_str = ""
                for x in node.input:
                    if x in in_weight_map:
                        input_str += f"{name_map[x]}({in_weight_map[x]}),"
                    else:
                        input_str += f"{name_map[x]},"
                node_rep += input_str[:-1]
            else:
                node_rep += "Const"

            if node.op_type in OP_DESCRIPTOR_FUNCS:
                node_rep += f"-->{','.join([name_map[x] for x in node.output])}:"
                node_rep += OP_DESCRIPTOR_FUNCS[node.op_type](node)
            else:
                node_rep += f"-->{','.join([name_map[x] for x in node.output])}:\n"
                node_rep += f"Op: {node.op_type}({','.join([f'{x.name}={self._attrtype_to_str(x)}' for x in node.attribute if self._attrtype_to_str(x) != '[]'])})"  # noqa: E501
                params = [
                    f"{name_map[x.name]}: {str(x.dims).replace(' ', '')}"
                    for x in self.onnx_model.graph.initializer
                    if x.name in node.input
                ]
                if len(params) > 0:
                    # node_rep += f"\n{'\n'.join(params)}"
                    node_rep += "\n" + "\n".join(params)
            str_rep.append(node_rep)
            node_idcs[node.name] = i
            for output in node.output:  # Make output of nodes refer back to the node
                node_idcs[output] = i
        for node in sorted(self.onnx_model.graph.output, key=lambda x: node_idcs[x.name]):
            str_rep.append(  # noqa: PERF401
                f"{name_map[node.name]}:\nOp: Output({tuple(x.dim_value for x in node.type.tensor_type.shape.dim)})",
            )
        return str_rep

    def _op_simplify(  # noqa: C901, PLR0912, PLR6301
        self,
        node: onnx.NodeProto,
        other_inputs: list[str],
        *,
        simp: bool = False,
    ) -> tuple[str, list[str]]:
        def attrtype_to_str(x: onnx.AttributeProto) -> str:  # noqa: C901, PLR0911, PLR0912
            match x.type:
                case onnx.AttributeProto.FLOAT:
                    return str(x.floats)
                case onnx.AttributeProto.FLOATS:
                    return str(x.floats).replace(" ", "")
                case onnx.AttributeProto.INT:
                    return str(x.ints)
                case onnx.AttributeProto.INTS:
                    # check if all elements are the same
                    if len(set(x.ints)) == 1:
                        return str(x.ints[0])
                    return str(x.ints).replace(" ", "")
                case onnx.AttributeProto.STRING:
                    return str(x.strings)
                case onnx.AttributeProto.STRINGS:
                    return str(x.strings).replace(" ", "")
                case onnx.AttributeProto.TENSOR:
                    return "<BLOB>"
                case onnx.AttributeProto.TENSORS:
                    return "<BLOB,...>"
                case onnx.AttributeProto.GRAPH:
                    return "<BLOB>"
                case onnx.AttributeProto.GRAPHS:
                    return "<BLOB,...>"
                case onnx.AttributeProto.SPARSE_TENSOR:
                    return "<BLOB>"
                case onnx.AttributeProto.SPARSE_TENSORS:
                    return "<BLOB,...>"
                case onnx.AttributeProto.TYPE_PROTO:
                    return "<TYPE>"
                case onnx.AttributeProto.TYPE_PROTOS:
                    return "<TYPE,...>"
                case onnx.AttributeProto.UNDEFINED:
                    return "undefined"
                case _:
                    msg = f"Unknown attribute type: {x.type}"
                    raise ValueError(msg)

        if not simp:
            return ",".join([
                f"{x.name}={attrtype_to_str(x)}" for x in node.attribute if attrtype_to_str(x) != "[]"
            ]), other_inputs

        match node.op_type:
            case "Conv":
                # only keep kernel_shape, remove stride, pads, dilations
                # remove all inputs, but infer channel number and bias from weight
                reduce_attribute = []
                for x in node.attribute:
                    if x.name == "kernel_shape":
                        reduce_attribute.append(f"{x.name}={attrtype_to_str(x)}")  # noqa: PERF401
                reduce_inputs = []
                for i in other_inputs:
                    if i.startswith("In"):
                        weight_shape = ast.literal_eval(i[2:])
                        if len(weight_shape) == 4:  # noqa: PLR2004
                            reduce_attribute.append(f"channels={weight_shape[1]}")
                        if len(weight_shape) == 1:
                            reduce_attribute.append("bias=True")
                return ",".join(reduce_attribute), reduce_inputs
            case "AveragePool" | "MaxPool":
                # only keep kernel_shape, remove stride, pads, dilations
                reduce_attribute = []
                for x in node.attribute:
                    if x.name == "kernel_shape":
                        reduce_attribute.append(f"{x.name}={attrtype_to_str(x)}")
                return ",".join(reduce_attribute), other_inputs
            case "BatchNormalization":
                # remove all inputs, but infer channel number
                reduce_attribute = []
                for i in other_inputs:
                    if i.startswith("In"):
                        weight_shape = ast.literal_eval(i[2:])
                        if len(weight_shape) != 1:
                            continue
                        reduce_attribute.append(f"channels={weight_shape[0]}")
                        break
                return ",".join(reduce_attribute), []
            case _:
                return ",".join([
                    f"{x.name}={attrtype_to_str(x)}" for x in node.attribute if attrtype_to_str(x) != "[]"
                ]), other_inputs
        # return ','.join([f'{x.name}={attrtype_to_str(x)}' for x in node.attribute if attrtype_to_str(x) != '[]']), other_inputs  # noqa: E501

    def _op_class(self, node: onnx.NodeProto) -> str:  # noqa: PLR6301
        POOL_TYPES = {  # noqa: N806
            "AveragePool": "avg",
            "MaxPool": "max",
            "GlobalAveragePool": "gobal_avg",
            "GlobalMaxPool": "global_max",
        }
        NORM_TYPES = {  # noqa: N806
            "BatchNormalization": "batch",
            "InstanceNormalization": "instance",
            "LayerNormalization": "layer",
        }
        ACT_TYPES = {  # noqa: N806
            "Relu": "relu",
            "LeakyRelu": "leaky_relu",
            "HardSwish": "hard_swish",
        }
        LINEAR_TYPES = {  # noqa: N806
            "Gemm": "gemm",
            "MatMul": "matmul",
        }
        SHAPE_TYPES = {  # noqa: N806
            "Reshape": "reshape",
            "Transpose": "transpose",
            "Unsqueeze": "unsqueeze",
        }

        if node.op_type in POOL_TYPES:
            return f"Pool({POOL_TYPES[node.op_type]})"
        if node.op_type in NORM_TYPES:
            return f"Norm({NORM_TYPES[node.op_type]})"
        if node.op_type in ACT_TYPES:
            return f"Act({ACT_TYPES[node.op_type]})"
        if node.op_type in LINEAR_TYPES:
            return f"Linear({LINEAR_TYPES[node.op_type]})"
        if node.op_type in SHAPE_TYPES:
            return f"Shape({SHAPE_TYPES[node.op_type]})"
        return node.op_type

    def get_onnx_infos_chain(self, *, unify_class: bool = False, simp_op: bool = False) -> str:  # noqa: C901, PLR0912, PLR0915
        """Chain consecutive nodes with single input and single output together"""  # noqa: DOC201
        shapes = {}
        for value_info in self.onnx_model.graph.value_info:
            shapes[value_info.name] = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
        for output in self.onnx_model.graph.output:
            shapes[output.name] = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        inputs = [input_.name for input_ in self.onnx_model.graph.input]
        params = [init.name for init in self.onnx_model.graph.initializer]

        name_map = {}

        for input_ in self.onnx_model.graph.input:
            name_map[input_.name] = f"In[{','.join([str(x.dim_value) for x in input_.type.tensor_type.shape.dim])}]"
        for init in self.onnx_model.graph.initializer:
            name_map[init.name] = f"Param{str(init.dims).replace(' ', '')}"
        for out in self.onnx_model.graph.output:
            name_map[out.name] = "Out"

        i = 1
        curr_str = ""
        curr_output = None
        for node in self.onnx_model.graph.node:
            node_inputs = [i for i in node.input if i not in inputs and i not in params]
            other_inputs = [name_map[i] for i in node.input if i in inputs or i in params]
            # print(node.name)
            attr, other_inputs = self._op_simplify(node, other_inputs, simp=simp_op)
            op_type = self._op_class(node) if unify_class else node.op_type
            if len(node_inputs) == 1 and len(node.output) == 1:
                # elegible for chain consecutive nodes
                if curr_output is None:
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0:  # print additional inputs (Param or Input)
                        if node_inputs[0] in name_map:
                            curr_str += f"{name_map[node_inputs[0]]}, {', '.join(other_inputs)})"
                        else:
                            curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += f"{name_map[node_inputs[0]]})" if node_inputs[0] in name_map else "prev)"
                    if attr != "":
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]
                elif curr_output in node_inputs:
                    # chain with previous node
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0:  # print additional inputs (Param or Input)
                        curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += "prev)"
                    if attr != "":
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]
                else:
                    # cannot chain
                    name_map[curr_output] = f"Value{i}"  # name the previous output
                    i += 1
                    if curr_output in shapes:
                        curr_str += f"{name_map[curr_output]}{shapes[curr_output]}\n"
                    else:
                        curr_str += f"{name_map[curr_output]}\n"
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0:  # print additional inputs (Param or Input)
                        if node_inputs[0] in name_map:
                            curr_str += f"{name_map[node_inputs[0]]}, {', '.join(other_inputs)})"
                        else:
                            curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += ")"
                    if attr != "":
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]

            else:
                # not elegible for chaining, print current output if exists
                if curr_output is not None:
                    name_map[curr_output] = f"Value{i}"
                    curr_str += f"{name_map[curr_output]}{shapes[curr_output]}"
                    i += 1
                    curr_str += "\n"

                curr_str += f"{node.op_type}("
                for input_ in node.input:
                    if input_ in name_map:
                        curr_str += f"{name_map[input_]}, "
                    else:
                        curr_str += "prev, "
                curr_str = curr_str[:-2]  # remove last comma and space
                if attr != "":
                    curr_str += f")({attr}) --> "
                else:
                    curr_str += ") --> "

                for out in node.output:
                    if out not in name_map:
                        name_map[out] = f"Value{i}"
                        i += 1
                curr_str += ", ".join([name_map[x] for x in node.output]) + f"{shapes[node.output[0]]}"
                if curr_str != "":
                    curr_str += "\n"
                curr_output = None
            # print(curr_str)

        if curr_output is not None:
            curr_str += name_map[curr_output]

        return curr_str

    def get_onnx_infos_chain_slim(self) -> str:  # noqa: C901, PLR0912, PLR0915
        """Chain consecutive nodes with single input and single output together"""  # noqa: DOC201

        def attrtype_to_str(x: onnx.AttributeProto) -> str:  # noqa: C901, PLR0911, PLR0912
            match x.type:
                case onnx.AttributeProto.FLOAT:
                    return str(x.floats)
                case onnx.AttributeProto.FLOATS:
                    return str(x.floats).replace(" ", "")
                case onnx.AttributeProto.INT:
                    return str(x.ints)
                case onnx.AttributeProto.INTS:
                    # check if all elements are the same
                    if len(set(x.ints)) == 1:
                        return str(x.ints[0])
                    return str(x.ints).replace(" ", "")
                case onnx.AttributeProto.STRING:
                    return str(x.strings)
                case onnx.AttributeProto.STRINGS:
                    return str(x.strings).replace(" ", "")
                case onnx.AttributeProto.TENSOR:
                    return "<BLOB>"
                case onnx.AttributeProto.TENSORS:
                    return "<BLOB,...>"
                case onnx.AttributeProto.GRAPH:
                    return "<BLOB>"
                case onnx.AttributeProto.GRAPHS:
                    return "<BLOB,...>"
                case onnx.AttributeProto.SPARSE_TENSOR:
                    return "<BLOB>"
                case onnx.AttributeProto.SPARSE_TENSORS:
                    return "<BLOB,...>"
                case onnx.AttributeProto.TYPE_PROTO:
                    return "<TYPE>"
                case onnx.AttributeProto.TYPE_PROTOS:
                    return "<TYPE,...>"
                case onnx.AttributeProto.UNDEFINED:
                    return "undefined"
                case _:
                    msg = f"Unknown attribute type: {x.type}"
                    raise ValueError(msg)

        shapes = {}
        for value_info in self.onnx_model.graph.value_info:
            shape_lsts = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            shapes[value_info.name] = "x".join([str(x) for x in shape_lsts])
        for output in self.onnx_model.graph.output:
            shape_lsts = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            shapes[output.name] = "x".join([str(x) for x in shape_lsts])
        inputs = [input_.name for input_ in self.onnx_model.graph.input]
        params = [init.name for init in self.onnx_model.graph.initializer]

        name_map = {}

        for input_ in self.onnx_model.graph.input:
            name_map[input_.name] = f"{'x'.join([str(x.dim_value) for x in input_.type.tensor_type.shape.dim])}"
        for init in self.onnx_model.graph.initializer:
            name_map[init.name] = f"Param{str(init.dims).replace(' ', '')}"
        for out in self.onnx_model.graph.output:
            name_map[out.name] = "Out"

        i = 1
        curr_str = ""
        curr_output = None
        for node in self.onnx_model.graph.node:
            node_inputs = [i for i in node.input if i not in inputs and i not in params]
            other_inputs = [name_map[i] for i in node.input if i in inputs or i in params]
            # print(node.name)
            attr = ",".join([f"{x.name}={attrtype_to_str(x)}" for x in node.attribute if attrtype_to_str(x) != "[]"])
            op_type = node.op_type
            if len(node_inputs) == 1 and len(node.output) == 1:
                # elegible for chain consecutive nodes
                if curr_output is None:
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0:  # print additional inputs (Param or Input)
                        if node_inputs[0] in name_map:
                            curr_str += f"{name_map[node_inputs[0]]}, {', '.join(other_inputs)})"
                        else:
                            curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += f"{name_map[node_inputs[0]]})" if node_inputs[0] in name_map else "prev)"
                    if attr != "":
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]
                elif curr_output in node_inputs:
                    # chain with previous node
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0:  # print additional inputs (Param or Input)
                        curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += "prev)"
                    if attr != "":
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]
                else:
                    # cannot chain
                    name_map[curr_output] = f"Value{i}"  # name the previous output
                    i += 1
                    if curr_output in shapes:
                        curr_str += f"{name_map[curr_output]}:{shapes[curr_output]}\n"
                    else:
                        curr_str += f"{name_map[curr_output]}\n"
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0:  # print additional inputs (Param or Input)
                        if node_inputs[0] in name_map:
                            curr_str += f"{name_map[node_inputs[0]]}, {', '.join(other_inputs)})"
                        else:
                            curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += ")"
                    if attr != "":
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]

            else:
                # not elegible for chaining, print current output if exists
                if curr_output is not None:
                    name_map[curr_output] = f"Value{i}"
                    curr_str += f"{name_map[curr_output]}:{shapes[curr_output]}"
                    i += 1
                    curr_str += "\n"

                curr_str += f"{node.op_type}("
                for input_ in node.input:
                    if input_ in name_map:
                        curr_str += f"{name_map[input_]}, "
                    else:
                        curr_str += "prev, "
                curr_str = curr_str[:-2]  # remove last comma and space
                if attr != "":
                    curr_str += f")({attr}) --> "
                else:
                    curr_str += ") --> "

                for out in node.output:
                    if out not in name_map:
                        name_map[out] = f"Value{i}"
                        i += 1
                curr_str += ", ".join([name_map[x] for x in node.output]) + f":{shapes[node.output[0]]}"
                if curr_str != "":
                    curr_str += "\n"
                curr_output = None
            # print(curr_str)

        if curr_output is not None:
            curr_str += name_map[curr_output]

        return curr_str

    def get_onnx_infos_chain_slim_base(self) -> str:  # noqa: C901, PLR0912
        """Chain consecutive nodes with single input and single output together"""  # noqa: DOC201
        shapes = {}
        for value_info in self.onnx_model.graph.value_info:
            shape_lsts = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            shapes[value_info.name] = "x".join([str(x) for x in shape_lsts])
        for output in self.onnx_model.graph.output:
            shape_lsts = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            shapes[output.name] = "x".join([str(x) for x in shape_lsts])
        inputs = [input_.name for input_ in self.onnx_model.graph.input]
        params = [init.name for init in self.onnx_model.graph.initializer]

        name_map = {}

        for input_ in self.onnx_model.graph.input:
            name_map[input_.name] = f"{'x'.join([str(x.dim_value) for x in input_.type.tensor_type.shape.dim])}"
        for init in self.onnx_model.graph.initializer:
            name_map[init.name] = f"Param{str(init.dims).replace(' ', '')}"
        for out in self.onnx_model.graph.output:
            name_map[out.name] = "Out"
        i = 1

        curr_str = ""
        curr_output = None
        for node in self.onnx_model.graph.node:
            node_inputs = [i for i in node.input if i not in inputs and i not in params]
            # other_inputs = [name_map[i] for i in node.input if i in inputs or i in params]
            # print(node.name)
            # attr = ",".join([f"{x.name}={attrtype_to_str(x)}" for x in node.attribute if attrtype_to_str(x) != "[]"])
            op_type = node.op_type
            if len(node_inputs) == 1 and len(node.output) == 1:
                # elegible for chain consecutive nodes
                if curr_output is None:
                    curr_str += f"{op_type}"
                    curr_str += " --> "
                    curr_output = node.output[0]
                elif curr_output in node_inputs:
                    # chain with previous node
                    curr_str += f"{op_type}"
                    curr_str += " --> "
                    curr_output = node.output[0]
                else:
                    # cannot chain
                    name_map[curr_output] = "Value"  # name the previous output
                    i += 1
                    curr_str += f"{name_map[curr_output]}\n"
                    curr_str += f"{op_type}"
                    curr_str += " --> "
                    curr_output = node.output[0]

            else:
                # not elegible for chaining, print current output if exists
                if curr_output is not None:
                    name_map[curr_output] = "Value"
                    curr_str += f"{name_map[curr_output]}"
                    i += 1
                    curr_str += "\n"

                curr_str += f"{node.op_type}"
                curr_str += " --> "

                for out in node.output:
                    if out not in name_map:
                        name_map[out] = "Value"
                        i += 1
                curr_str += ", ".join([name_map[x] for x in node.output])
                if curr_str != "":
                    curr_str += "\n"
                curr_output = None
            # print(curr_str)

        if curr_output is not None:
            curr_str += name_map[curr_output]

        return curr_str

    def get_onnx_infos_chain_slim_input(self) -> str:  # noqa: C901, PLR0912, PLR0915
        """Chain consecutive nodes with single input and single output together"""  # noqa: DOC201
        shapes = {}
        for value_info in self.onnx_model.graph.value_info:
            shape_lsts = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            shapes[value_info.name] = "x".join([str(x) for x in shape_lsts])
        for output in self.onnx_model.graph.output:
            shape_lsts = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            shapes[output.name] = "x".join([str(x) for x in shape_lsts])
        inputs = [input_.name for input_ in self.onnx_model.graph.input]
        params = [init.name for init in self.onnx_model.graph.initializer]

        name_map = {}

        for input_ in self.onnx_model.graph.input:
            name_map[input_.name] = f"{'x'.join([str(x.dim_value) for x in input_.type.tensor_type.shape.dim])}"
        for init in self.onnx_model.graph.initializer:
            name_map[init.name] = f"Param{str(init.dims).replace(' ', '')}"
        for out in self.onnx_model.graph.output:
            name_map[out.name] = "Out"
        i = 1

        curr_str = ""
        curr_output = None
        for node in self.onnx_model.graph.node:
            node_inputs = [i for i in node.input if i not in inputs and i not in params]
            other_inputs = [name_map[i] for i in node.input if i in inputs or i in params]
            # print(node.name)
            # attr = ",".join([f"{x.name}={attrtype_to_str(x)}" for x in node.attribute if attrtype_to_str(x) != "[]"])
            op_type = node.op_type
            if len(node_inputs) == 1 and len(node.output) == 1:
                # elegible for chain consecutive nodes
                if curr_output is None:
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0:  # print additional inputs (Param or Input)
                        if node_inputs[0] in name_map:
                            curr_str += f"{name_map[node_inputs[0]]}, {', '.join(other_inputs)})"
                        else:
                            curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += f"{name_map[node_inputs[0]]})" if node_inputs[0] in name_map else "prev)"
                    curr_str += " --> "
                    curr_output = node.output[0]
                elif curr_output in node_inputs:
                    # chain with previous node
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0:  # print additional inputs (Param or Input)
                        curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += "prev)"
                    curr_str += " --> "
                    curr_output = node.output[0]
                else:
                    # cannot chain
                    name_map[curr_output] = f"Value{i}"  # name the previous output
                    i += 1
                    if curr_output in shapes:
                        curr_str += f"{name_map[curr_output]}\n"
                    else:
                        curr_str += f"{name_map[curr_output]}\n"
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0:  # print additional inputs (Param or Input)
                        if node_inputs[0] in name_map:
                            curr_str += f"{name_map[node_inputs[0]]}, {', '.join(other_inputs)})"
                        else:
                            curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += ")"
                    curr_str += " --> "
                    curr_output = node.output[0]

            else:
                # not elegible for chaining, print current output if exists
                if curr_output is not None:
                    name_map[curr_output] = f"Value{i}"
                    curr_str += f"{name_map[curr_output]}"
                    i += 1
                    curr_str += "\n"

                curr_str += f"{node.op_type}("
                for input_ in node.input:
                    if input_ in name_map:
                        curr_str += f"{name_map[input_]}, "
                    else:
                        curr_str += "prev, "
                curr_str = curr_str[:-2]  # remove last comma and space
                curr_str += ") --> "

                for out in node.output:
                    if out not in name_map:
                        name_map[out] = f"Value{i}"
                        i += 1
                curr_str += ", ".join([name_map[x] for x in node.output])
                if curr_str != "":
                    curr_str += "\n"
                curr_output = None
            # print(curr_str)

        if curr_output is not None:
            curr_str += name_map[curr_output]

        return curr_str

    def get_onnx_infos_chain_slim_param(self) -> str:  # noqa: C901, PLR0912, PLR0915
        """Chain consecutive nodes with single input and single output together"""  # noqa: DOC201

        def attrtype_to_str(x: onnx.AttributeProto) -> str:  # noqa: C901, PLR0911, PLR0912
            match x.type:
                case onnx.AttributeProto.FLOAT:
                    return str(x.floats)
                case onnx.AttributeProto.FLOATS:
                    return str(x.floats).replace(" ", "")
                case onnx.AttributeProto.INT:
                    return str(x.ints)
                case onnx.AttributeProto.INTS:
                    # check if all elements are the same
                    if len(set(x.ints)) == 1:
                        return str(x.ints[0])
                    return str(x.ints).replace(" ", "")
                case onnx.AttributeProto.STRING:
                    return str(x.strings)
                case onnx.AttributeProto.STRINGS:
                    return str(x.strings).replace(" ", "")
                case onnx.AttributeProto.TENSOR:
                    return "<BLOB>"
                case onnx.AttributeProto.TENSORS:
                    return "<BLOB,...>"
                case onnx.AttributeProto.GRAPH:
                    return "<BLOB>"
                case onnx.AttributeProto.GRAPHS:
                    return "<BLOB,...>"
                case onnx.AttributeProto.SPARSE_TENSOR:
                    return "<BLOB>"
                case onnx.AttributeProto.SPARSE_TENSORS:
                    return "<BLOB,...>"
                case onnx.AttributeProto.TYPE_PROTO:
                    return "<TYPE>"
                case onnx.AttributeProto.TYPE_PROTOS:
                    return "<TYPE,...>"
                case onnx.AttributeProto.UNDEFINED:
                    return "undefined"
                case _:
                    msg = f"Unknown attribute type: {x.type}"
                    raise ValueError(msg)

        shapes = {}
        for value_info in self.onnx_model.graph.value_info:
            shape_lsts = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            shapes[value_info.name] = "x".join([str(x) for x in shape_lsts])
        for output in self.onnx_model.graph.output:
            shape_lsts = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            shapes[output.name] = "x".join([str(x) for x in shape_lsts])
        inputs = [input_.name for input_ in self.onnx_model.graph.input]
        params = [init.name for init in self.onnx_model.graph.initializer]

        name_map = {}

        for input_ in self.onnx_model.graph.input:
            name_map[input_.name] = f"{'x'.join([str(x.dim_value) for x in input_.type.tensor_type.shape.dim])}"
        for init in self.onnx_model.graph.initializer:
            name_map[init.name] = f"Param{str(init.dims).replace(' ', '')}"
        for out in self.onnx_model.graph.output:
            name_map[out.name] = "Out"
        i = 1

        curr_str = ""
        curr_output = None
        for node in self.onnx_model.graph.node:
            node_inputs = [i for i in node.input if i not in inputs and i not in params]
            # other_inputs = [name_map[i] for i in node.input if i in inputs or i in params]
            # print(node.name)
            attr = ",".join([f"{x.name}={attrtype_to_str(x)}" for x in node.attribute if attrtype_to_str(x) != "[]"])
            op_type = node.op_type
            if len(node_inputs) == 1 and len(node.output) == 1:
                # elegible for chain consecutive nodes
                if curr_output is None:
                    curr_str += f"{op_type}"
                    if attr != "":
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]
                elif curr_output in node_inputs:
                    # chain with previous node
                    curr_str += f"{op_type}"
                    if attr != "":
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]
                else:
                    # cannot chain
                    name_map[curr_output] = "Value"  # name the previous output
                    i += 1
                    if curr_output in shapes:
                        curr_str += f"{name_map[curr_output]}\n"
                    else:
                        curr_str += f"{name_map[curr_output]}\n"
                    curr_str += f"{op_type}"
                    if attr != "":
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]

            else:
                # not elegible for chaining, print current output if exists
                if curr_output is not None:
                    name_map[curr_output] = "Value"
                    curr_str += f"{name_map[curr_output]}"
                    i += 1
                    curr_str += "\n"

                curr_str += f"{node.op_type}"
                if attr != "":
                    curr_str += f"({attr}) --> "
                else:
                    curr_str += " --> "

                for out in node.output:
                    if out not in name_map:
                        name_map[out] = "Value"
                        i += 1
                curr_str += ", ".join([name_map[x] for x in node.output])
                if curr_str != "":
                    curr_str += "\n"
                curr_output = None
            # print(curr_str)

        if curr_output is not None:
            curr_str += name_map[curr_output]

        return curr_str

    def get_onnx_infos_chain_slim_outshape(self) -> str:  # noqa: C901, PLR0912, PLR0915
        """Chain consecutive nodes with single input and single output together"""  # noqa: DOC201
        shapes = {}
        for value_info in self.onnx_model.graph.value_info:
            shape_lsts = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            shapes[value_info.name] = "x".join([str(x) for x in shape_lsts])
        for output in self.onnx_model.graph.output:
            shape_lsts = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            shapes[output.name] = "x".join([str(x) for x in shape_lsts])
        inputs = [input_.name for input_ in self.onnx_model.graph.input]
        params = [init.name for init in self.onnx_model.graph.initializer]

        name_map = {}

        for input_ in self.onnx_model.graph.input:
            name_map[input_.name] = f"{'x'.join([str(x.dim_value) for x in input_.type.tensor_type.shape.dim])}"
        for init in self.onnx_model.graph.initializer:
            name_map[init.name] = f"Param{str(init.dims).replace(' ', '')}"
        for out in self.onnx_model.graph.output:
            name_map[out.name] = "Out"
        i = 1

        curr_str = ""
        curr_output = None
        for node in self.onnx_model.graph.node:
            node_inputs = [i for i in node.input if i not in inputs and i not in params]
            # other_inputs = [name_map[i] for i in node.input if i in inputs or i in params]
            # print(node.name)
            # attr = ",".join([f"{x.name}={attrtype_to_str(x)}" for x in node.attribute if attrtype_to_str(x) != "[]"])
            op_type = node.op_type
            if len(node_inputs) == 1 and len(node.output) == 1:
                # elegible for chain consecutive nodes
                if curr_output is None:
                    curr_str += f"{op_type}"
                    curr_str += " --> "
                    curr_output = node.output[0]
                elif curr_output in node_inputs:
                    # chain with previous node
                    curr_str += f"{op_type}"
                    curr_str += " --> "
                    curr_output = node.output[0]
                else:
                    # cannot chain
                    name_map[curr_output] = "Value"  # name the previous output
                    i += 1
                    if curr_output in shapes:
                        curr_str += f"{name_map[curr_output]}:{shapes[curr_output]}\n"
                    else:
                        curr_str += f"{name_map[curr_output]}\n"
                    curr_str += f"{op_type}"
                    curr_str += " --> "
                    curr_output = node.output[0]

            else:
                # not elegible for chaining, print current output if exists
                if curr_output is not None:
                    name_map[curr_output] = "Value"
                    curr_str += f"{name_map[curr_output]}:{shapes[curr_output]}"
                    i += 1
                    curr_str += "\n"

                curr_str += f"{node.op_type}"
                curr_str += ") --> "

                for out in node.output:
                    if out not in name_map:
                        name_map[out] = "Value"
                        i += 1
                curr_str += ", ".join([name_map[x] for x in node.output]) + f":{shapes[node.output[0]]}"
                if curr_str != "":
                    curr_str += "\n"
                curr_output = None
            # print(curr_str)

        if curr_output is not None:
            curr_str += name_map[curr_output]

        return curr_str

    def get_onnx_infos_chain_nooutshape(self, *, unify_class: bool = False, simp_op: bool = False) -> str:  # noqa: C901, PLR0912, PLR0915
        """Chain consecutive nodes with single input and single output together"""  # noqa: DOC201
        shapes = {}
        for value_info in self.onnx_model.graph.value_info:
            shapes[value_info.name] = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
        inputs = [input_.name for input_ in self.onnx_model.graph.input]
        params = [init.name for init in self.onnx_model.graph.initializer]

        name_map = {}

        for input_ in self.onnx_model.graph.input:
            name_map[input_.name] = f"In[{','.join([str(x.dim_value) for x in input_.type.tensor_type.shape.dim])}]"
        for init in self.onnx_model.graph.initializer:
            name_map[init.name] = f"Param{str(init.dims).replace(' ', '')}"
        for out in self.onnx_model.graph.output:
            name_map[out.name] = f"Out[{','.join([str(x.dim_value) for x in out.type.tensor_type.shape.dim])}]"
        i = 1

        curr_str = ""
        curr_output = None
        for node in self.onnx_model.graph.node:
            node_inputs = [i for i in node.input if i not in inputs and i not in params]
            other_inputs = [name_map[i] for i in node.input if i in inputs or i in params]
            # print(node.name)
            attr, other_inputs = self._op_simplify(node, other_inputs, simp=simp_op)
            op_type = self._op_class(node) if unify_class else node.op_type
            if len(node_inputs) == 1 and len(node.output) == 1:
                # elegible for chain consecutive nodes
                if curr_output is None:
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0:  # print additional inputs (Param or Input)
                        if node_inputs[0] in name_map:
                            curr_str += f"{name_map[node_inputs[0]]}, {', '.join(other_inputs)})"
                        else:
                            curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += f"{name_map[node_inputs[0]]})" if node_inputs[0] in name_map else "prev)"
                    if attr != "":
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]
                elif curr_output in node_inputs:
                    # chain with previous node
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0:  # print additional inputs (Param or Input)
                        curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += "prev)"
                    if attr != "":
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]
                else:
                    # cannot chain
                    name_map[curr_output] = f"Value{i}"  # name the previous output
                    i += 1
                    curr_str += f"{name_map[curr_output]}\n"
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0:  # print additional inputs (Param or Input)
                        if node_inputs[0] in name_map:
                            curr_str += f"{name_map[node_inputs[0]]}, {', '.join(other_inputs)})"
                        else:
                            curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += ")"
                    if attr != "":
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]

            else:
                # not elegible for chaining, print current output if exists
                if curr_output is not None:
                    name_map[curr_output] = f"Value{i}"
                    curr_str += f"{name_map[curr_output]}"
                    i += 1
                    curr_str += "\n"

                curr_str += f"{node.op_type}("
                for input_ in node.input:
                    if input_ in name_map:
                        curr_str += f"{name_map[input_]}, "
                    else:
                        curr_str += "prev, "
                curr_str = curr_str[:-2]  # remove last comma and space
                if attr != "":
                    curr_str += f")({attr}) --> "
                else:
                    curr_str += ") --> "

                for out in node.output:
                    if out not in name_map:
                        name_map[out] = f"Value{i}"
                        i += 1
                curr_str += ", ".join([name_map[x] for x in node.output])
                if curr_str != "":
                    curr_str += "\n"
                curr_output = None
            # print(curr_str)

        if curr_output is not None:
            curr_str += name_map[curr_output]

        return curr_str

    def get_onnx_infos_minimum(self, *, incl_inputs: bool = False, incl_out_shape: bool = False) -> str:  # noqa: C901

        def attrtype_to_str(x: onnx.AttributeProto) -> str:  # noqa: C901, PLR0911, PLR0912
            match x.type:
                case onnx.AttributeProto.FLOAT:
                    return str(x.floats)
                case onnx.AttributeProto.FLOATS:
                    return str(x.floats).replace(" ", "")
                case onnx.AttributeProto.INT:
                    return str(x.ints)
                case onnx.AttributeProto.INTS:
                    # check if all elements are the same
                    if len(set(x.ints)) == 1:
                        return str(x.ints[0])
                    return str(x.ints).replace(" ", "")
                case onnx.AttributeProto.STRING:
                    return str(x.strings)
                case onnx.AttributeProto.STRINGS:
                    return str(x.strings).replace(" ", "")
                case onnx.AttributeProto.TENSOR:
                    return "<BLOB>"
                case onnx.AttributeProto.TENSORS:
                    return "<BLOB,...>"
                case onnx.AttributeProto.GRAPH:
                    return "<BLOB>"
                case onnx.AttributeProto.GRAPHS:
                    return "<BLOB,...>"
                case onnx.AttributeProto.SPARSE_TENSOR:
                    return "<BLOB>"
                case onnx.AttributeProto.SPARSE_TENSORS:
                    return "<BLOB,...>"
                case onnx.AttributeProto.TYPE_PROTO:
                    return "<TYPE>"
                case onnx.AttributeProto.TYPE_PROTOS:
                    return "<TYPE,...>"
                case onnx.AttributeProto.UNDEFINED:
                    return "undefined"
                case _:
                    msg = f"Unknown attribute type: {x.type}"
                    raise ValueError(msg)

        shapes = {}
        for value_info in self.onnx_model.graph.value_info:
            shapes[value_info.name] = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]

        name_map = {}

        for input_ in self.onnx_model.graph.input:
            name_map[input_.name] = f"[{','.join([str(x.dim_value) for x in input_.type.tensor_type.shape.dim])}]"
        for init in self.onnx_model.graph.initializer:
            name_map[init.name] = f"{str(init.dims).replace(' ', '')}"
        for out in self.onnx_model.graph.output:
            name_map[out.name] = f"[{','.join([str(x.dim_value) for x in out.type.tensor_type.shape.dim])}]"

        curr_str = ""
        for node in self.onnx_model.graph.node:
            if not incl_inputs:
                curr_str += f"{node.op_type}"
            else:
                curr_str += f"{node.op_type}("
                for input_ in node.input:
                    if input_ in name_map:
                        curr_str += f"{name_map[input_]}, "
                    else:
                        curr_str += "prev, "
                curr_str = curr_str[:-2]  # remove last comma and space
                curr_str += ")"
            attr = ",".join([
                f"{x.name}={attrtype_to_str(x)}" for x in node.attribute if self._attrtype_to_str(x) != "[]"
            ])
            if attr != "":
                curr_str += f"({attr})"
            if incl_out_shape:
                curr_str += " --> "
                curr_str += f"{shapes[node.output[0]]}" if node.output[0] in shapes else "[]"
            curr_str += "\n"
        return curr_str

    def get_onnx_infos_minimum_slim(self, *, incl_inputs: bool = False, incl_out_shape: bool = False) -> str:  # noqa: C901

        def attrtype_to_str(x: onnx.AttributeProto) -> str:  # noqa: C901, PLR0911, PLR0912
            match x.type:
                case onnx.AttributeProto.FLOAT:
                    return str(x.floats)
                case onnx.AttributeProto.FLOATS:
                    return str(x.floats).replace(" ", "")
                case onnx.AttributeProto.INT:
                    return str(x.ints)
                case onnx.AttributeProto.INTS:
                    # check if all elements are the same
                    if len(set(x.ints)) == 1:
                        return str(x.ints[0])
                    return str(x.ints).replace(" ", "")
                case onnx.AttributeProto.STRING:
                    return str(x.strings)
                case onnx.AttributeProto.STRINGS:
                    return str(x.strings).replace(" ", "")
                case onnx.AttributeProto.TENSOR:
                    return "<BLOB>"
                case onnx.AttributeProto.TENSORS:
                    return "<BLOB,...>"
                case onnx.AttributeProto.GRAPH:
                    return "<BLOB>"
                case onnx.AttributeProto.GRAPHS:
                    return "<BLOB,...>"
                case onnx.AttributeProto.SPARSE_TENSOR:
                    return "<BLOB>"
                case onnx.AttributeProto.SPARSE_TENSORS:
                    return "<BLOB,...>"
                case onnx.AttributeProto.TYPE_PROTO:
                    return "<TYPE>"
                case onnx.AttributeProto.TYPE_PROTOS:
                    return "<TYPE,...>"
                case onnx.AttributeProto.UNDEFINED:
                    return "undefined"
                case _:
                    msg = f"Unknown attribute type: {x.type}"
                    raise ValueError(msg)

        shapes = {}
        for value_info in self.onnx_model.graph.value_info:
            shape_lsts = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            shapes[value_info.name] = "x".join([str(x) for x in shape_lsts])

        name_map = {}

        for input_ in self.onnx_model.graph.input:
            name_map[input_.name] = f"{'x'.join([str(x.dim_value) for x in input_.type.tensor_type.shape.dim])}"
        for init in self.onnx_model.graph.initializer:
            name_map[init.name] = f"{str(init.dims).replace(' ', '')}"
        for out in self.onnx_model.graph.output:
            name_map[out.name] = f"{'x'.join([str(x.dim_value) for x in out.type.tensor_type.shape.dim])}"

        curr_str = ""
        for node in self.onnx_model.graph.node:
            if not incl_inputs:
                curr_str += f"{node.op_type}"
            else:
                curr_str += f"{node.op_type}("
                for input_ in node.input:
                    if input_ in name_map:
                        curr_str += f"{name_map[input_]}, "
                    else:
                        curr_str += "prev, "
                curr_str = curr_str[:-2]  # remove last comma and space
                curr_str += ")"
            attr = ",".join([
                f"{x.name}={attrtype_to_str(x)}" for x in node.attribute if self._attrtype_to_str(x) != "[]"
            ])
            if attr != "":
                curr_str += f"({attr})"
            if incl_out_shape:
                curr_str += " --> "
                curr_str += f"{shapes[node.output[0]]}" if node.output[0] in shapes else "[]"
            curr_str += "\n"
        return curr_str

    def get_onnx_str(
        self,
        mode: Literal[
            "full",
            "slim",
            "slim_noidentity",
            "oponly",
            "oponly_extended",
            "chain",
            "template",
            "template_slim",
            "chain_nooutshape",
            "chain_slim",
            "minimum",
            "chain_slim_base",
            "chain_slim_param",
            "chain_slim_outshape",
            "chain_slim_input",
        ] = "full",
        **args: Any,  # noqa: ANN401
    ) -> tuple[str, float, int]:
        mode_map = {
            "full": self.get_onnx_infos,
            "slim": self.get_onnx_infos_slim,
            "slim_noidentity": self.get_onnx_infos_slim_noidentity,
            "oponly": self.get_onnx_infos_oponly,
            "oponly_extended": self.get_onnx_infos_oponly_extended,
            "chain": self.get_onnx_infos_chain,
            "template": self.get_onnx_infos_template_compressed,
            "template_slim": self.get_onnx_infos_template_compressed_slim,
            "chain_nooutshape": self.get_onnx_infos_chain_nooutshape,
            "chain_slim": self.get_onnx_infos_chain_slim,
            "minimum": self.get_onnx_infos_minimum,
            "chain_slim_base": self.get_onnx_infos_chain_slim_base,
            "chain_slim_param": self.get_onnx_infos_chain_slim_param,
            "chain_slim_outshape": self.get_onnx_infos_chain_slim_outshape,
            "chain_slim_input": self.get_onnx_infos_chain_slim_input,
        }

        onnx_nodes = mode_map[mode](**args)
        if isinstance(onnx_nodes, list):
            onnx_nodes = "\n".join(onnx_nodes)
        token_count = self._get_token_count(onnx_nodes, self.tokenizer)
        acc = float(self.onnx_model.metadata_props[0].value)
        return onnx_nodes, acc, token_count


def get_node_attribute(node, attr_name, default=None):
    """Extracts an attribute value from an ONNX node."""
    for attr in node.attribute:
        if attr.name == attr_name:
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            elif attr.type == onnx.AttributeProto.STRING:
                return attr.s.decode('utf-8')
    return default

def format_node_op(node):
    """
    Converts a single ONNX node into a concise function call string.
    Example: Conv(c=32, k=3, s=1)
    """
    op_type = node.op_type
    
    # 1. Convolution
    if op_type == 'Conv':
        # Try to infer out_channels from weights if possible, usually not easy from node alone
        # without looking at weight tensor. We rely on shape inference for channels mostly,
        # but let's grab kernel/stride/pad.
        k = get_node_attribute(node, 'kernel_shape', [])
        s = get_node_attribute(node, 'strides', [])
        p = get_node_attribute(node, 'pads', [])
        d = get_node_attribute(node, 'dilations', [])
        
        # Simplify formatting: only show if relevant
        args = []
        if k: args.append(f"k={k[0]}x{k[1]}" if len(k)>1 else f"k={k}")
        if s and any(x != 1 for x in s): args.append(f"s={s[0]}") # Assume square usually
        if p and any(x != 0 for x in p): args.append(f"p={p[0]}")
        if d and any(x != 1 for x in d): args.append(f"d={d[0]}")
        
        return f"Conv({', '.join(args)})"

    # 2. Pooling
    elif 'Pool' in op_type: # MaxPool, AveragePool, GlobalAveragePool
        k = get_node_attribute(node, 'kernel_shape', [])
        s = get_node_attribute(node, 'strides', [])
        args = []
        if k: args.append(f"k={k[0]}")
        if s and any(x != 1 for x in s): args.append(f"s={s[0]}")
        return f"{op_type}({', '.join(args)})"
    
    # 3. Activations (no params usually)
    elif op_type in ['Relu', 'Sigmoid', 'Tanh', 'Softmax']:
        return f"{op_type}"

    # 4. Merges
    elif op_type == 'Concat':
        axis = get_node_attribute(node, 'axis', 1)
        return f"Concat(axis={axis})"
    
    # 5. Generic Fallback
    else:
        return op_type

def onnx_to_nas_string(model_path, tokenizer: PreTrainedTokenizer):
    # 1. Load and Infer Shapes
    model = onnx.load(model_path)
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    
    # Map: tensor_name -> Shape tuple (e.g. (1, 32, 64, 64))
    value_shapes = {}
    for val in graph.value_info:
        dims = [d.dim_value for d in val.type.tensor_type.shape.dim]
        value_shapes[val.name] = dims
    # Also get input/output shapes
    for val in graph.input:
        dims = [d.dim_value for d in val.type.tensor_type.shape.dim]
        value_shapes[val.name] = dims
    for val in graph.output:
        dims = [d.dim_value for d in val.type.tensor_type.shape.dim]
        value_shapes[val.name] = dims

    # 2. Build Graph Connectivity (Adjacency List)
    # We must ignore Initializers (weights/biases) in the input count to detect "linear" chains.
    initializers = set(t.name for t in graph.initializer)
    
    node_map = {node.name : node for node in graph.node} # Map name to node
    # Since nodes might not have names, we use index or assume topological order (ONNX standard).
    # We will iterate list directly.
    
    # Map: tensor_name -> producing_node_index
    tensor_producer = {}
    # Map: node_index -> list of input_node_indices (Data inputs only)
    node_parents = {i: [] for i in range(len(graph.node))}
    node_children = {i: [] for i in range(len(graph.node))}

    for i, node in enumerate(graph.node):
        # Record outputs
        for out_name in node.output:
            tensor_producer[out_name] = i
    
    # Link parents/children
    for i, node in enumerate(graph.node):
        for inp_name in node.input:
            if inp_name in initializers: continue # Skip weights
            if inp_name in tensor_producer:
                parent_idx = tensor_producer[inp_name]
                node_parents[i].append(parent_idx)
                node_children[parent_idx].append(i)

    # 3. Chain Collapsing Algorithm
    visited = [False] * len(graph.node)
    execution_order = [] # List of strings/blocks

    for i in range(len(graph.node)):
        if visited[i]: continue
        
        current_idx = i
        chain = []
        
        # Walk the chain
        while True:
            node = graph.node[current_idx]
            visited[current_idx] = True
            chain.append(current_idx)
            
            # Check if we can continue chaining
            # Condition: Current node has exactly 1 child, and that child has exactly 1 parent (Current)
            children = node_children[current_idx]
            
            if len(children) == 1:
                child_idx = children[0]
                parents_of_child = node_parents[child_idx]
                
                if len(parents_of_child) == 1:
                    # It's a linear link! Continue.
                    current_idx = child_idx
                else:
                    # Child is a merge node (e.g. Concat of 2 branches), stop here.
                    break
            else:
                # Branching point (0 children or >1 children), stop here.
                break
        
        # Process the chain we just found
        # Determine Input Name for this block
        first_node = graph.node[chain[0]]
        # Find the "Data" input (not weight)
        data_inputs = [x for x in first_node.input if x not in initializers]
        input_name = data_inputs[0] if data_inputs else "GlobalInput"
        
        # Determine Output Shape
        last_node = graph.node[chain[-1]]
        output_name = last_node.output[0]
        out_shape = value_shapes.get(output_name, "?")
        
        # Format the block
        if len(chain) > 1:
            # Sequential Block
            ops_strs = [format_node_op(graph.node[x]) for x in chain]
            block_str = f"x_{chain[-1]} = Sequential([\n    " + ",\n    ".join(ops_strs) + f"\n])(x_{tensor_producer.get(input_name, 'in')})"
        else:
            # Single Node
            # Logic for Merge nodes (Concat) needs to explicitly list inputs
            if first_node.op_type == 'Concat':
                # Convert input tensor names to variable names like x_12, x_44
                inputs_formatted = []
                for inp in data_inputs:
                    prod_idx = tensor_producer.get(inp)
                    inputs_formatted.append(f"x_{prod_idx}" if prod_idx is not None else "input")
                block_str = f"x_{chain[-1]} = {format_node_op(first_node)}([{', '.join(inputs_formatted)}])"
            else:
                # Standard single node
                prev_idx = tensor_producer.get(input_name)
                prev_var = f"x_{prev_idx}" if prev_idx is not None else "input"
                block_str = f"x_{chain[-1]} = {format_node_op(first_node)}({prev_var})"

        # Append Shape info as comment
        block_str += f" # -> {out_shape}"
        execution_order.append(block_str)

    str_encoding = "\n".join(execution_order)
    token_count = len(tokenizer.tokenize(str_encoding, add_special_tokens=False))
    acc = float(model.metadata_props[0].value)
    return str_encoding, acc, token_count

def get_node_attribute(node, attr_name, default=None):
    for attr in node.attribute:
        if attr.name == attr_name:
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            elif attr.type == onnx.AttributeProto.STRING:
                return attr.s.decode('utf-8')
    return default

def format_shape(shape_list):
    if not shape_list: return "?"
    dims = shape_list[1:] if len(shape_list) > 1 else shape_list
    return "*".join(str(d) for d in dims)

def format_compact_op(node, value_shapes, flow_inputs):
    op_type = node.op_type
    alias_map = {
        'Conv': 'C', 'BatchNormalization': 'BN', 'Relu': 'R', 'Add': 'Add', 
        'Concat': 'Cat', 'GlobalAveragePool': 'GAP', 'AveragePool': 'AvgP', 
        'MaxPool': 'P', 'Softmax': 'Softmax', 'Gemm': 'FC', 'MatMul': 'MM', 
        'Flatten': 'Flat', 'Reshape': 'Reshape', 'Transpose': 'Transpose'
    }
    short_op = alias_map.get(op_type, op_type)
    args = []

    # 1. Standard Attributes
    if op_type in ['Conv', 'MaxPool', 'AveragePool', 'GlobalAveragePool']:
        k = get_node_attribute(node, 'kernel_shape')
        s = get_node_attribute(node, 'strides')
        p = get_node_attribute(node, 'pads')
        g = get_node_attribute(node, 'group', 1)
        
        if k: 
            val = k[0] if (len(k)==1 or k[0]==k[1]) else f"{k[0]}x{k[1]}"
            args.append(f"k={val}")
        if s and any(x != 1 for x in s):
            val = s[0] if (len(s)==1 or s[0]==s[1]) else f"{s[0]}x{s[1]}"
            args.append(f"s={val}")
        if p and any(x != 0 for x in p):
            args.append(f"p={p[0]}")
        if g and g > 1:
            args.append(f"g={g}")

    # 2. Parameter Attributes (Weights disguised as inputs)
    # We check for inputs that are NOT in the flow_inputs list.
    all_inputs = node.input
    param_inputs = [x for x in all_inputs if x not in flow_inputs and x != '']
    
    if param_inputs:
        if op_type in ['MatMul', 'Gemm']:
            # Usually input 1 is weight
            w_name = param_inputs[0]
            w_shape = value_shapes.get(w_name)
            if w_shape:
                w_str = "x".join(str(x) for x in w_shape)
                args.append(f"W={w_str}")
        elif op_type == 'BatchNormalization':
            # BN params are boring (channels), usually redundant with input shape.
            # We ignore them to save tokens.
            pass

    if args:
        return f"{short_op}({', '.join(args)})"
    else:
        return short_op

def onnx_to_dense_string(model_path, tokenizer: PreTrainedTokenizer):
    # --- 1. Load and Infer Shapes ---
    model = onnx.load(model_path)
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    
    value_shapes = {}
    def register_shapes(proto_list):
        for val in proto_list:
            if hasattr(val, 'type'): 
                dims = [d.dim_value for d in val.type.tensor_type.shape.dim]
                value_shapes[val.name] = dims
            elif hasattr(val, 'dims'):
                value_shapes[val.name] = list(val.dims)

    register_shapes(graph.value_info)
    register_shapes(graph.input)
    register_shapes(graph.output)
    register_shapes(graph.initializer)

    # --- 2. Identify The "True" Data Input ---
    # Heuristic: The first input in the list is the Data (Image/Text).
    # All other inputs in graph.input are likely Weights/Biases (BN stats, etc).
    if not graph.input:
        main_input_name = "global_input" # Fallback
    else:
        main_input_name = graph.input[0].name
    
    # --- 3. Build Connectivity ---
    tensor_producer = {}
    for i, node in enumerate(graph.node):
        for out_name in node.output:
            tensor_producer[out_name] = i

    # [FIX]: A "Flow Edge" is strictly:
    # 1. Output from another Node
    # 2. OR The Main Graph Input
    # Everything else (Initializers, Secondary Graph Inputs) is a Parameter.
    def is_flow_edge(inp_name):
        is_node_output = inp_name in tensor_producer
        is_main_input = inp_name == main_input_name
        return is_node_output or is_main_input

    node_children = {i: [] for i in range(len(graph.node))}
    node_parents = {i: [] for i in range(len(graph.node))}

    for i, node in enumerate(graph.node):
        flow_inputs = [inp for inp in node.input if is_flow_edge(inp)]
        for inp_name in flow_inputs:
            if inp_name in tensor_producer:
                parent_idx = tensor_producer[inp_name]
                node_parents[i].append(parent_idx)
                node_children[parent_idx].append(i)

    # --- 4. Chain Collapsing ---
    tensor_to_logical_id = {}
    tensor_to_logical_id[main_input_name] = 0 # Map main input to 0

    logical_counter = 1
    visited = [False] * len(graph.node)
    execution_lines = []

    for i in range(len(graph.node)):
        if visited[i]: continue
        
        chain = []
        curr = i
        
        # Linear Chain Walk
        while True:
            visited[curr] = True
            chain.append(curr)
            
            children = node_children[curr]
            if len(children) == 1:
                child_idx = children[0]
                if len(node_parents[child_idx]) == 1:
                    curr = child_idx
                else:
                    break
            else:
                break

        # --- Format Block ---
        first_node = graph.node[chain[0]]
        last_node = graph.node[chain[-1]]
        
        # Determine Inputs for this block
        # Only consider FLOW inputs.
        raw_inputs = first_node.input
        valid_input_ids = []
        flow_input_names = [] 
        
        for inp in raw_inputs:
            if not is_flow_edge(inp):
                continue # Skip weights (both Initializers AND Secondary Graph Inputs)

            if inp in tensor_to_logical_id:
                valid_input_ids.append(str(tensor_to_logical_id[inp]))
                flow_input_names.append(inp)
            elif inp == main_input_name:
                valid_input_ids.append("0")
                flow_input_names.append(inp)
        
        if not valid_input_ids:
             input_str = "Const"
        else:
             input_str = ", ".join(valid_input_ids)

        ops_strs = []
        
        # Handling Merge vs Linear Start
        if first_node.op_type in ['Concat', 'Add'] and len(valid_input_ids) > 1:
             op_text = format_compact_op(first_node, value_shapes, flow_input_names)
             short_op = 'Cat' if first_node.op_type == 'Concat' else 'Add'
             ops_strs.append(f"{short_op}({input_str})")
             input_str = "" 
        else:
             # Passing flow_input_names allows format_compact_op to see what was skipped
             # and print it as a parameter (e.g. W=...)
             ops_strs.append(format_compact_op(first_node, value_shapes, flow_input_names))

        # Rest of Chain
        for node_idx in chain[1:]:
            node_obj = graph.node[node_idx]
            # Implicit flow from previous node in chain
            prev_flow = [node_parents[node_idx][0]] if node_parents[node_idx] else []
            ops_strs.append(format_compact_op(node_obj, value_shapes, prev_flow))

        prefix = f"{input_str} > " if input_str else ""
        chain_str = prefix + " > ".join(ops_strs)
        
        out_name = last_node.output[0]
        out_shape = value_shapes.get(out_name, [])
        shape_str = format_shape(out_shape)
        
        line = f"{logical_counter}: {chain_str} | {shape_str}"
        execution_lines.append(line)

        for out in last_node.output:
            tensor_to_logical_id[out] = logical_counter
        
        logical_counter += 1

    str_encoding = "\n".join(execution_lines)
    token_count = len(tokenizer.tokenize(str_encoding, add_special_tokens=False))
    acc = float(model.metadata_props[0].value)
    return str_encoding, acc, token_count