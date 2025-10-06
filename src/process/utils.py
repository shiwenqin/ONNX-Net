import os
import ast
from collections import defaultdict, deque
from functools import partial

import onnx
import pandas as pd
import numpy as np
import numpy.typing as npt
import networkx as nx
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizer

from onnx_graph_utils import onnx_to_graph, encode_graph

def list_onnx_files(folder_path):
    """
    Return a list of all .onnx file paths found within 'folder_path' (recursively).
    """
    onnx_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.onnx'):
                full_path = os.path.join(root, file)
                onnx_files.append(full_path)
    return onnx_files

class ONNXConverter:
    '''
    Class contains multiple methods to convert ONNX model to string representation.
    The one used in the paper is get_onnx_str(mode ='chain_slim').

    Variants in ablation study can be found with:
    - get_onnx_str(mode ='chain_slim_base')
    - get_onnx_str(mode ='chain_slim_param')
    - get_onnx_str(mode ='chain_slim_outshape')
    - get_onnx_str(mode ='chain_slim_input')
    '''

    def __init__(self, onnx_path: str, tokenizer: PreTrainedTokenizer):
        self.onnx_model = onnx.load(onnx_path)
        self.tokenizer = tokenizer

        self.onnx_model = onnx.shape_inference.infer_shapes(self.onnx_model)
        #self.onnx_model = onnx.version_converter.convert_version(self.onnx_model, 18)

    def _attrtype_to_str(self, x: onnx.AttributeProto) -> str:
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
                assert False

    def _get_token_count(self, model_str: str, tokenizer: PreTrainedTokenizer) -> int:
        return len(tokenizer.tokenize(model_str, add_special_tokens=False))

    def get_onnx_infos(self) -> tuple[list[str], npt.NDArray[np.uint64]]:
        # Generate new names for input-/output-/intermediate values
        def next_name() -> str:
            ret = f"Param{next_name.i}"  # pyright: ignore [reportFunctionMemberAccess]
            next_name.i += 1  # pyright: ignore [reportFunctionMemberAccess]
            return ret

        next_name.i = 1  # pyright: ignore [reportFunctionMemberAccess]
        i = 1
        name_map = defaultdict(next_name)
        for node in self.onnx_model.graph.input:
            name_map[node.name] = f"In{i}"
            i += 1
        i = 1
        for node in self.onnx_model.graph.node:
            for output in node.output:
                name_map[output] = f"Value{i}"
                i += 1
        i = 1
        for node in self.onnx_model.graph.output:
            name_map[node.name] = f"Out{i}"
            i += 1

        # Generate string representation of the model and node indices in the adjacency matrix
        node_idcs = {}
        str_rep = []
        for i, node in enumerate(self.onnx_model.graph.input):
            node_idcs[node.name] = i
            str_rep.append(
                f"{name_map[node.name]}:\nOp: Input({tuple(x.dim_value for x in node.type.tensor_type.shape.dim)})"
            )
        for i, node in enumerate(self.onnx_model.graph.node, max(node_idcs.values()) + 1):
            node_rep = ""
            node_rep += f"{','.join([name_map[x] for x in node.input]) if len([name_map[x] for x in node.input]) > 0 else 'Const'}-->{','.join([name_map[x] for x in node.output])}:\n"
            node_rep += f"Op: {node.op_type}({','.join([f'{x.name}={self._attrtype_to_str(x)}' for x in node.attribute if self._attrtype_to_str(x) != '[]'])})"
            params = [
                f"{name_map[x.name]}: {str(x.dims).replace(' ', '')}"
                for x in self.onnx_model.graph.initializer
                if x.name in node.input
            ]
            if len(params) > 0:
                # node_rep += f"\n{'\n'.join(params)}"
                node_rep += "\n"+ '\n'.join(params)

            str_rep.append(node_rep)
            node_idcs[node.name] = i
            for output in node.output:  # Make output of nodes refer back to the node
                node_idcs[output] = i
        for i, node in enumerate(sorted(self.onnx_model.graph.output, key=lambda x: node_idcs[x.name])):
            str_rep.append(f"{name_map[node.name]}:\nOp: Output({tuple(x.dim_value for x in node.type.tensor_type.shape.dim)})")

        return str_rep
    
    def get_onnx_infos_slim(self) -> tuple[list[str], npt.NDArray[np.uint64]]:
        '''
        Remove Input section, merge the input shape into Op section.
        '''

        # Generate new names for input-/output-/intermediate values
        def next_name() -> str:
            ret = f"Param{next_name.i}"  # pyright: ignore [reportFunctionMemberAccess]
            next_name.i += 1  # pyright: ignore [reportFunctionMemberAccess]
            return ret

        next_name.i = 1  # pyright: ignore [reportFunctionMemberAccess]
        i = 1
        name_map = defaultdict(next_name)
        for node in self.onnx_model.graph.input:
            name_map[node.name] = f"In{i}"
            i += 1
        i = 1
        for node in self.onnx_model.graph.node:
            for output in node.output:
                name_map[output] = f"Value{i}"
                i += 1
        i = 1
        for node in self.onnx_model.graph.output:
            name_map[node.name] = f"Out{i}"
            i += 1

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
                input_part = 'Const'

            output_names = [name_map[x] for x in node.output]
            output_part = ','.join(output_names)

            node_rep += f"{input_part}-->{output_part}:\n"
            node_rep += f"Op: {node.op_type}({','.join([f'{x.name}={self._attrtype_to_str(x)}' for x in node.attribute if self._attrtype_to_str(x) != '[]'])})"
            params = [
                f"{name_map[x.name]}: {str(x.dims).replace(' ', '')}"
                for x in self.onnx_model.graph.initializer
                if x.name in node.input
            ]
            if len(params) > 0:
                # node_rep += f"\n{'\n'.join(params)}"
                node_rep += "\n"+ '\n'.join(params)

            str_rep.append(node_rep)
            node_idcs[node.name] = i
            for output in node.output:  # Make output of nodes refer back to the node
                node_idcs[output] = i
        for i, node in enumerate(sorted(self.onnx_model.graph.output, key=lambda x: node_idcs[x.name])):
            str_rep.append(f"{name_map[node.name]}:\nOp: Output({tuple(x.dim_value for x in node.type.tensor_type.shape.dim)})")
        
        return str_rep
    
    def get_onnx_infos_slim_noidentity(self) -> tuple[list[str], npt.NDArray[np.uint64]]:
        # Generate new names for input-/output-/intermediate values
        def next_name() -> str:
            ret = f"Param{next_name.i}"  # pyright: ignore [reportFunctionMemberAccess]
            next_name.i += 1  # pyright: ignore [reportFunctionMemberAccess]
            return ret

        next_name.i = 1  # pyright: ignore [reportFunctionMemberAccess]
        i = 1
        name_map = defaultdict(next_name)
        for node in self.onnx_model.graph.input:
            name_map[node.name] = f"In{i}"
            i += 1
        i = 1
        for node in self.onnx_model.graph.node:
            for output in node.output:
                name_map[output] = f"Value{i}"
                i += 1
        i = 1
        for node in self.onnx_model.graph.output:
            name_map[node.name] = f"Out{i}"
            i += 1

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
                input_part = 'Const'

            if node.op_type == "Identity" and node.input[0] in in_weight_map:
                for x in node.output:
                    in_weight_map[x] = in_weight_map[node.input[0]]
            else:
                output_names = [name_map[x] for x in node.output]
                output_part = ','.join(output_names)

                node_rep += f"{input_part}-->{output_part}:\n"
                node_rep += f"Op: {node.op_type}({','.join([f'{x.name}={self._attrtype_to_str(x)}' for x in node.attribute if self._attrtype_to_str(x) != '[]'])})"
                params = [
                    f"{name_map[x.name]}: {str(x.dims).replace(' ', '')}"
                    for x in self.onnx_model.graph.initializer
                    if x.name in node.input
                ]
                if len(params) > 0:
                    # node_rep += f"\n{'\n'.join(params)}"
                    node_rep += "\n"+ '\n'.join(params)

                str_rep.append(node_rep)
                node_idcs[node.name] = i
                for output in node.output:  # Make output of nodes refer back to the node
                    node_idcs[output] = i
        for i, node in enumerate(sorted(self.onnx_model.graph.output, key=lambda x: node_idcs[x.name])):
            str_rep.append(f"{name_map[node.name]}:\nOp: Output({tuple(x.dim_value for x in node.type.tensor_type.shape.dim)})")
        
        return str_rep

    def get_onnx_infos_oponly(self) -> tuple[list[str], npt.NDArray[np.uint64]]:
        node_idcs = {}
        str_rep = []
        for i, node in enumerate(self.onnx_model.graph.node):
            node_rep = ""
            node_rep += f"Op: {node.op_type}({','.join([f'{x.name}={self._attrtype_to_str(x)}' for x in node.attribute if self._attrtype_to_str(x) != '[]'])})"
            str_rep.append(node_rep)
            node_idcs[node.name] = i
            for output in node.output:  # Make output of nodes refer back to the node
                node_idcs[output] = i
        for i, node in enumerate(sorted(self.onnx_model.graph.output, key=lambda x: node_idcs[x.name])):
            str_rep.append(f"Op: Output({tuple(x.dim_value for x in node.type.tensor_type.shape.dim)})")
        
        return str_rep

    def get_onnx_infos_oponly_extended(self) -> tuple[list[str], npt.NDArray[np.uint64]]:
        node_idcs = {}
        str_rep = []
        for i, node in enumerate(self.onnx_model.graph.node):
            node_rep = ""
            node_rep += f"Op: {node.op_type}({','.join([f'{x.name}={self._attrtype_to_str(x)}' for x in node.attribute if self._attrtype_to_str(x) != '[]'])})"
            node_rep += f", Inputs: {len(node.input)}, Outputs: {len(node.output)}"
            str_rep.append(node_rep)
            node_idcs[node.name] = i
            for output in node.output:  # Make output of nodes refer back to the node
                node_idcs[output] = i
        for i, node in enumerate(sorted(self.onnx_model.graph.output, key=lambda x: node_idcs[x.name])):
            str_rep.append(f"Op: Output({tuple(x.dim_value for x in node.type.tensor_type.shape.dim)})")
        
        return str_rep

    def get_onnx_infos_template_compressed(
            self,
        ) -> tuple[list[str], npt.NDArray[np.uint64]]:
            # Generate new names for input-/output-/intermediate values
            def next_name() -> str:
                ret = f"Param{next_name.i}"  # pyright: ignore [reportFunctionMemberAccess]
                next_name.i += 1  # pyright: ignore [reportFunctionMemberAccess]
                return ret

            next_name.i = 1  # pyright: ignore [reportFunctionMemberAccess]
            i = 1
            name_map = defaultdict(next_name)
            for node in self.onnx_model.graph.input:
                name_map[node.name] = f"In{i}"
                i += 1
            i = 1
            for node in self.onnx_model.graph.node:
                for output in node.output:
                    name_map[output] = f"Value{i}"
                    i += 1
            i = 1
            for node in self.onnx_model.graph.output:
                name_map[node.name] = f"Out{i}"
                i += 1

            OP_DESCRIPTOR_FUNCS = {
                "Input": lambda _: "IN()",
                "Constant": lambda _: "CON()",
                "Output": lambda _: "OUT()",
                "Conv": lambda n: "C("
                + str(next(filter(lambda x: x.name == "kernel_shape", n.attribute)).ints[0])
                + ","
                + str(next(filter(lambda x: x.name == "strides", n.attribute)).ints[0])
                + ","
                + str(next(filter(lambda x: x.name == "pads", n.attribute)).ints[0])
                + ")",
                "Relu": lambda _: "R()",
                "Add": lambda _: "A()",
                "MaxPool": lambda n: f"MP({str(next(filter(lambda x: x.name == 'kernel_shape', n.attribute)).ints[0])})",
                "Concat": lambda _: "CC()",
                "MatMul": lambda _: "MM()",
                "Gemm": lambda _: "MM()",
                "BatchNormalization": lambda _: "BN()",
                "AveragePool": lambda n: f"AP({str(next(filter(lambda x: x.name == 'kernel_shape', n.attribute)).ints[0])})",
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
                + "Unkown operations are printed in a generic format with full names and attributes."
            ]
            for i, node in enumerate(self.onnx_model.graph.input):
                node_idcs[node.name] = i
                str_rep.append(
                    f"{name_map[node.name]}:\nOp: Input({tuple(x.dim_value for x in node.type.tensor_type.shape.dim)})"
                )
            for i, node in enumerate(
                self.onnx_model.graph.node, max(node_idcs.values()) + 1
            ):
                node_rep = ""
                if node.op_type in OP_DESCRIPTOR_FUNCS:
                    node_rep += f"{','.join([name_map[x] for x in node.input]) if len([name_map[x] for x in node.input]) > 0 else 'Const'}-->{','.join([name_map[x] for x in node.output])}:"
                    node_rep += OP_DESCRIPTOR_FUNCS[node.op_type](node)
                else:
                    node_rep += f"{','.join([name_map[x] for x in node.input]) if len([name_map[x] for x in node.input]) > 0 else 'Const'}-->{','.join([name_map[x] for x in node.output])}:\n"
                    node_rep += f"Op: {node.op_type}({','.join([f'{x.name}={self._attrtype_to_str(x)}' for x in node.attribute if self._attrtype_to_str(x) != '[]'])})"
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
            for i, node in enumerate(
                sorted(self.onnx_model.graph.output, key=lambda x: node_idcs[x.name])
            ):
                str_rep.append(
                    f"{name_map[node.name]}:\nOp: Output({tuple(x.dim_value for x in node.type.tensor_type.shape.dim)})")
            return str_rep

    def get_onnx_infos_template_compressed_slim(self):

        # Generate new names for input-/output-/intermediate values
            def next_name() -> str:
                ret = f"Param{next_name.i}"  # pyright: ignore [reportFunctionMemberAccess]
                next_name.i += 1  # pyright: ignore [reportFunctionMemberAccess]
                return ret

            next_name.i = 1  # pyright: ignore [reportFunctionMemberAccess]
            i = 1
            name_map = defaultdict(next_name)
            for node in self.onnx_model.graph.input:
                name_map[node.name] = f"In{i}"
                i += 1
            i = 1
            for node in self.onnx_model.graph.node:
                for output in node.output:
                    name_map[output] = f"Value{i}"
                    i += 1
            i = 1
            for node in self.onnx_model.graph.output:
                name_map[node.name] = f"Out{i}"
                i += 1

            OP_DESCRIPTOR_FUNCS = {
                "Input": lambda _: "IN()",
                "Constant": lambda _: "CON()",
                "Output": lambda _: "OUT()",
                "Conv": lambda n: "C("
                + str(next(filter(lambda x: x.name == "kernel_shape", n.attribute)).ints[0])
                + ","
                + str(next(filter(lambda x: x.name == "strides", n.attribute)).ints[0])
                + ","
                + str(next(filter(lambda x: x.name == "pads", n.attribute)).ints[0])
                + ")",
                "Relu": lambda _: "R()",
                "Add": lambda _: "A()",
                "MaxPool": lambda n: f"MP({str(next(filter(lambda x: x.name == 'kernel_shape', n.attribute)).ints[0])})",
                "Concat": lambda _: "CC()",
                "MatMul": lambda _: "MM()",
                "Gemm": lambda _: "MM()",
                "BatchNormalization": lambda _: "BN()",
                "AveragePool": lambda n: f"AP({str(next(filter(lambda x: x.name == 'kernel_shape', n.attribute)).ints[0])})",
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
                + "Unkown operations are printed in a generic format with full names and attributes."
            ]
            for i, node in enumerate(self.onnx_model.graph.input):
                node_idcs[node.name] = i
                in_weight_map[node.name] = ",".join([str(x.dim_value) for x in node.type.tensor_type.shape.dim])
            for i, node in enumerate(
                self.onnx_model.graph.node, max(node_idcs.values()) + 1
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
                    node_rep += 'Const'

                if node.op_type in OP_DESCRIPTOR_FUNCS:
                    node_rep += f"-->{','.join([name_map[x] for x in node.output])}:"
                    node_rep += OP_DESCRIPTOR_FUNCS[node.op_type](node)
                else:
                    node_rep += f"-->{','.join([name_map[x] for x in node.output])}:\n"
                    node_rep += f"Op: {node.op_type}({','.join([f'{x.name}={self._attrtype_to_str(x)}' for x in node.attribute if self._attrtype_to_str(x) != '[]'])})"
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
            for i, node in enumerate(
                sorted(self.onnx_model.graph.output, key=lambda x: node_idcs[x.name])
            ):
                str_rep.append(
                    f"{name_map[node.name]}:\nOp: Output({tuple(x.dim_value for x in node.type.tensor_type.shape.dim)})")
            return str_rep

    def _op_simplify(self, node: onnx.NodeProto, other_inputs: list[str], simp: bool = False) -> str:
        def attrtype_to_str(x: onnx.AttributeProto) -> str:
            match x.type:
                case onnx.AttributeProto.FLOAT:
                    return str(x.floats)
                case onnx.AttributeProto.FLOATS:
                    return str(x.floats).replace(" ", "")
                case onnx.AttributeProto.INT:
                    return str(x.ints)
                case onnx.AttributeProto.INTS:
                    # check if all elements are the same
                    if len(set(list(x.ints))) == 1:
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
                    assert False
        
        if not simp:
            return ','.join([f'{x.name}={attrtype_to_str(x)}' for x in node.attribute if attrtype_to_str(x) != '[]']), other_inputs
        
        match node.op_type:
            case "Conv":
                # only keep kernel_shape, remove stride, pads, dilations
                # remove all inputs, but infer channel number and bias from weight
                reduce_attribute = []
                for x in node.attribute:
                    if x.name in ["kernel_shape"]:
                        reduce_attribute.append(f"{x.name}={attrtype_to_str(x)}")
                reduce_inputs = []
                for i in other_inputs:
                    if i.startswith("In"):
                        weight_shape = ast.literal_eval(i[2:])
                        if len(weight_shape) == 4:
                            reduce_attribute.append(f"channels={weight_shape[1]}")
                        if len(weight_shape) == 1:
                            reduce_attribute.append("bias=True")
                return ','.join(reduce_attribute), reduce_inputs
            case "AveragePool" | "MaxPool":
                # only keep kernel_shape, remove stride, pads, dilations
                reduce_attribute = []
                for x in node.attribute:
                    if x.name in ["kernel_shape"]:
                        reduce_attribute.append(f"{x.name}={attrtype_to_str(x)}")
                return ','.join(reduce_attribute), other_inputs
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
                return ','.join(reduce_attribute), []
            case _:
                return ','.join([f'{x.name}={attrtype_to_str(x)}' for x in node.attribute if attrtype_to_str(x) != '[]']), other_inputs
        #return ','.join([f'{x.name}={attrtype_to_str(x)}' for x in node.attribute if attrtype_to_str(x) != '[]']), other_inputs

    def _op_class(self, node: onnx.NodeProto) -> str:
        
        POOL_TYPES = {
            "AveragePool": "avg",
            "MaxPool": "max",
            "GlobalAveragePool": "gobal_avg",
            "GlobalMaxPool": "global_max"
        }
        NORM_TYPES = {
            "BatchNormalization": "batch",
            "InstanceNormalization": "instance",
            "LayerNormalization": "layer"
        }
        ACT_TYPES = {
            "Relu": "relu",
            "LeakyRelu": "leaky_relu",
            "HardSwish": "hard_swish",
        }
        LINEAR_TYPES = {
            "Gemm": "gemm",
            "MatMul": "matmul",
        }
        SHAPE_TYPES = {
            "Reshape": "reshape",
            "Transpose": "transpose",
            "Unsqueeze": "unsqueeze"
        }

        if node.op_type in POOL_TYPES:
            return f"Pool({POOL_TYPES[node.op_type]})"
        elif node.op_type in NORM_TYPES:
            return f"Norm({NORM_TYPES[node.op_type]})"
        elif node.op_type in ACT_TYPES:
            return f"Act({ACT_TYPES[node.op_type]})"
        elif node.op_type in LINEAR_TYPES:
            return f"Linear({LINEAR_TYPES[node.op_type]})"
        elif node.op_type in SHAPE_TYPES:
            return f"Shape({SHAPE_TYPES[node.op_type]})"
        else:
            return node.op_type

    def get_onnx_infos_chain(self, unify_class=False, simp_op=False) -> str:
        '''
        Chain consecutive nodes with single input and single output together
        '''

        shapes = {}
        for value_info in self.onnx_model.graph.value_info:
            shapes[value_info.name] = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
        for output in self.onnx_model.graph.output:
            shapes[output.name] = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        inputs = [input.name for input in self.onnx_model.graph.input]
        params = [init.name for init in self.onnx_model.graph.initializer]

        name_map = {}

        i = 1
        for input in self.onnx_model.graph.input:
            name_map[input.name] = f"In[{','.join([str(x.dim_value) for x in input.type.tensor_type.shape.dim])}]"
            i += 1
        i = 1
        for init in self.onnx_model.graph.initializer:
            name_map[init.name] = f"Param{str(init.dims).replace(' ', '')}"
            i += 1
        i = 1
        for out in self.onnx_model.graph.output:
            name_map[out.name] = "Out"
            i += 1
        i = 1

        curr_str = ""
        curr_output = None
        for node in self.onnx_model.graph.node:
            node_inputs = [i for i in node.input if i not in inputs and i not in params]
            other_inputs = [name_map[i] for i in node.input if i in inputs or i in params]
            #print(node.name)
            attr, other_inputs = self._op_simplify(node, other_inputs, simp=simp_op)
            op_type = self._op_class(node) if unify_class else node.op_type
            if len(node_inputs) == 1 and len(node.output) == 1:
                # elegible for chain consecutive nodes
                if curr_output is None:
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0: # print additional inputs (Param or Input)
                        if node_inputs[0] in name_map:
                            curr_str += f"{name_map[node_inputs[0]]}, {', '.join(other_inputs)})"
                        else:
                            curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += f"{name_map[node_inputs[0]]})" if node_inputs[0] in name_map else "prev)"
                    if attr != '':
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]
                elif curr_output in node_inputs:
                    # chain with previous node
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0: # print additional inputs (Param or Input)
                        curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += "prev)"
                    if attr != '':
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]
                else:
                    # cannot chain
                    name_map[curr_output] = f"Value{i}" # name the previous output
                    i += 1
                    if curr_output in shapes:
                        curr_str += f"{name_map[curr_output]}{shapes[curr_output]}\n"
                    else:
                        curr_str += f"{name_map[curr_output]}\n"
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0: # print additional inputs (Param or Input)
                        if node_inputs[0] in name_map:
                            curr_str += f"{name_map[node_inputs[0]]}, {', '.join(other_inputs)})"
                        else:
                            curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += ")"
                    if attr != '':
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
                    curr_str += '\n'

                curr_str += f"{node.op_type}("
                for input in node.input:
                    if input in name_map:
                        curr_str += f"{name_map[input]}, "
                    else:
                        curr_str += "prev, "
                curr_str = curr_str[:-2]  # remove last comma and space
                if attr != '':
                    curr_str += f")({attr}) --> "
                else:
                    curr_str += ") --> "

                for out in node.output:
                    if out not in name_map:
                        name_map[out] = f"Value{i}"
                        i += 1
                curr_str += ", ".join([name_map[x] for x in node.output]) + f"{shapes[node.output[0]]}"
                if curr_str != "": 
                    curr_str += '\n'
                curr_output = None
            #print(curr_str)
                
        if curr_output is not None:
            curr_str += name_map[curr_output]

        return curr_str

    def get_onnx_infos_chain_slim(self) -> str:
        '''
        Chain consecutive nodes with single input and single output together
        '''
        def attrtype_to_str(x: onnx.AttributeProto) -> str:
            match x.type:
                case onnx.AttributeProto.FLOAT:
                    return str(x.floats)
                case onnx.AttributeProto.FLOATS:
                    return str(x.floats).replace(" ", "")
                case onnx.AttributeProto.INT:
                    return str(x.ints)
                case onnx.AttributeProto.INTS:
                    # check if all elements are the same
                    if len(set(list(x.ints))) == 1:
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
                    assert False

        shapes = {}
        for value_info in self.onnx_model.graph.value_info:
            shape_lsts = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            shapes[value_info.name] = 'x'.join([str(x) for x in shape_lsts])
        for output in self.onnx_model.graph.output:
            shape_lsts = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            shapes[output.name] = 'x'.join([str(x) for x in shape_lsts])
        inputs = [input.name for input in self.onnx_model.graph.input]
        params = [init.name for init in self.onnx_model.graph.initializer]

        name_map = {}

        i = 1
        for input in self.onnx_model.graph.input:
            name_map[input.name] = f"{'x'.join([str(x.dim_value) for x in input.type.tensor_type.shape.dim])}"
            i += 1
        i = 1
        for init in self.onnx_model.graph.initializer:
            name_map[init.name] = f"Param{str(init.dims).replace(' ', '')}"
            i += 1
        i = 1
        for out in self.onnx_model.graph.output:
            name_map[out.name] = "Out"
            i += 1
        i = 1

        curr_str = ""
        curr_output = None
        for node in self.onnx_model.graph.node:
            node_inputs = [i for i in node.input if i not in inputs and i not in params]
            other_inputs = [name_map[i] for i in node.input if i in inputs or i in params]
            #print(node.name)
            attr = ','.join([f'{x.name}={attrtype_to_str(x)}' for x in node.attribute if attrtype_to_str(x) != '[]'])
            op_type = node.op_type
            if len(node_inputs) == 1 and len(node.output) == 1:
                # elegible for chain consecutive nodes
                if curr_output is None:
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0: # print additional inputs (Param or Input)
                        if node_inputs[0] in name_map:
                            curr_str += f"{name_map[node_inputs[0]]}, {', '.join(other_inputs)})"
                        else:
                            curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += f"{name_map[node_inputs[0]]})" if node_inputs[0] in name_map else "prev)"
                    if attr != '':
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]
                elif curr_output in node_inputs:
                    # chain with previous node
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0: # print additional inputs (Param or Input)
                        curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += "prev)"
                    if attr != '':
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]
                else:
                    # cannot chain
                    name_map[curr_output] = f"Value{i}" # name the previous output
                    i += 1
                    if curr_output in shapes:
                        curr_str += f"{name_map[curr_output]}:{shapes[curr_output]}\n"
                    else:
                        curr_str += f"{name_map[curr_output]}\n"
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0: # print additional inputs (Param or Input)
                        if node_inputs[0] in name_map:
                            curr_str += f"{name_map[node_inputs[0]]}, {', '.join(other_inputs)})"
                        else:
                            curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += ")"
                    if attr != '':
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
                    curr_str += '\n'

                curr_str += f"{node.op_type}("
                for input in node.input:
                    if input in name_map:
                        curr_str += f"{name_map[input]}, "
                    else:
                        curr_str += "prev, "
                curr_str = curr_str[:-2]  # remove last comma and space
                if attr != '':
                    curr_str += f")({attr}) --> "
                else:
                    curr_str += ") --> "

                for out in node.output:
                    if out not in name_map:
                        name_map[out] = f"Value{i}"
                        i += 1
                curr_str += ", ".join([name_map[x] for x in node.output]) + f":{shapes[node.output[0]]}"
                if curr_str != "": 
                    curr_str += '\n'
                curr_output = None
            #print(curr_str)
                
        if curr_output is not None:
            curr_str += name_map[curr_output]

        return curr_str

    def get_onnx_infos_chain_slim_base(self) -> str:
        '''
        Chain consecutive nodes with single input and single output together
        '''
        def attrtype_to_str(x: onnx.AttributeProto) -> str:
            match x.type:
                case onnx.AttributeProto.FLOAT:
                    return str(x.floats)
                case onnx.AttributeProto.FLOATS:
                    return str(x.floats).replace(" ", "")
                case onnx.AttributeProto.INT:
                    return str(x.ints)
                case onnx.AttributeProto.INTS:
                    # check if all elements are the same
                    if len(set(list(x.ints))) == 1:
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
                    assert False

        shapes = {}
        for value_info in self.onnx_model.graph.value_info:
            shape_lsts = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            shapes[value_info.name] = 'x'.join([str(x) for x in shape_lsts])
        for output in self.onnx_model.graph.output:
            shape_lsts = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            shapes[output.name] = 'x'.join([str(x) for x in shape_lsts])
        inputs = [input.name for input in self.onnx_model.graph.input]
        params = [init.name for init in self.onnx_model.graph.initializer]

        name_map = {}

        i = 1
        for input in self.onnx_model.graph.input:
            name_map[input.name] = f"{'x'.join([str(x.dim_value) for x in input.type.tensor_type.shape.dim])}"
            i += 1
        i = 1
        for init in self.onnx_model.graph.initializer:
            name_map[init.name] = f"Param{str(init.dims).replace(' ', '')}"
            i += 1
        i = 1
        for out in self.onnx_model.graph.output:
            name_map[out.name] = "Out"
            i += 1
        i = 1

        curr_str = ""
        curr_output = None
        for node in self.onnx_model.graph.node:
            node_inputs = [i for i in node.input if i not in inputs and i not in params]
            other_inputs = [name_map[i] for i in node.input if i in inputs or i in params]
            #print(node.name)
            attr = ','.join([f'{x.name}={attrtype_to_str(x)}' for x in node.attribute if attrtype_to_str(x) != '[]'])
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
                    name_map[curr_output] = f"Value" # name the previous output
                    i += 1
                    curr_str += f"{name_map[curr_output]}\n"
                    curr_str += f"{op_type}"
                    curr_str += " --> "
                    curr_output = node.output[0]

            else:
                # not elegible for chaining, print current output if exists
                if curr_output is not None:
                    name_map[curr_output] = f"Value"
                    curr_str += f"{name_map[curr_output]}"
                    i += 1
                    curr_str += '\n'

                curr_str += f"{node.op_type}"
                curr_str += " --> "

                for out in node.output:
                    if out not in name_map:
                        name_map[out] = f"Value"
                        i += 1
                curr_str += ", ".join([name_map[x] for x in node.output])
                if curr_str != "": 
                    curr_str += '\n'
                curr_output = None
            #print(curr_str)
                
        if curr_output is not None:
            curr_str += name_map[curr_output]

        return curr_str

    def get_onnx_infos_chain_slim_input(self) -> str:
        '''
        Chain consecutive nodes with single input and single output together
        '''
        def attrtype_to_str(x: onnx.AttributeProto) -> str:
            match x.type:
                case onnx.AttributeProto.FLOAT:
                    return str(x.floats)
                case onnx.AttributeProto.FLOATS:
                    return str(x.floats).replace(" ", "")
                case onnx.AttributeProto.INT:
                    return str(x.ints)
                case onnx.AttributeProto.INTS:
                    # check if all elements are the same
                    if len(set(list(x.ints))) == 1:
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
                    assert False

        shapes = {}
        for value_info in self.onnx_model.graph.value_info:
            shape_lsts = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            shapes[value_info.name] = 'x'.join([str(x) for x in shape_lsts])
        for output in self.onnx_model.graph.output:
            shape_lsts = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            shapes[output.name] = 'x'.join([str(x) for x in shape_lsts])
        inputs = [input.name for input in self.onnx_model.graph.input]
        params = [init.name for init in self.onnx_model.graph.initializer]

        name_map = {}

        i = 1
        for input in self.onnx_model.graph.input:
            name_map[input.name] = f"{'x'.join([str(x.dim_value) for x in input.type.tensor_type.shape.dim])}"
            i += 1
        i = 1
        for init in self.onnx_model.graph.initializer:
            name_map[init.name] = f"Param{str(init.dims).replace(' ', '')}"
            i += 1
        i = 1
        for out in self.onnx_model.graph.output:
            name_map[out.name] = "Out"
            i += 1
        i = 1

        curr_str = ""
        curr_output = None
        for node in self.onnx_model.graph.node:
            node_inputs = [i for i in node.input if i not in inputs and i not in params]
            other_inputs = [name_map[i] for i in node.input if i in inputs or i in params]
            #print(node.name)
            attr = ','.join([f'{x.name}={attrtype_to_str(x)}' for x in node.attribute if attrtype_to_str(x) != '[]'])
            op_type = node.op_type
            if len(node_inputs) == 1 and len(node.output) == 1:
                # elegible for chain consecutive nodes
                if curr_output is None:
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0: # print additional inputs (Param or Input)
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
                    if len(other_inputs) > 0: # print additional inputs (Param or Input)
                        curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += "prev)"
                    curr_str += " --> "
                    curr_output = node.output[0]
                else:
                    # cannot chain
                    name_map[curr_output] = f"Value{i}" # name the previous output
                    i += 1
                    if curr_output in shapes:
                        curr_str += f"{name_map[curr_output]}\n"
                    else:
                        curr_str += f"{name_map[curr_output]}\n"
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0: # print additional inputs (Param or Input)
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
                    curr_str += '\n'

                curr_str += f"{node.op_type}("
                for input in node.input:
                    if input in name_map:
                        curr_str += f"{name_map[input]}, "
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
                    curr_str += '\n'
                curr_output = None
            #print(curr_str)
                
        if curr_output is not None:
            curr_str += name_map[curr_output]

        return curr_str

    def get_onnx_infos_chain_slim_param(self) -> str:
        '''
        Chain consecutive nodes with single input and single output together
        '''
        def attrtype_to_str(x: onnx.AttributeProto) -> str:
            match x.type:
                case onnx.AttributeProto.FLOAT:
                    return str(x.floats)
                case onnx.AttributeProto.FLOATS:
                    return str(x.floats).replace(" ", "")
                case onnx.AttributeProto.INT:
                    return str(x.ints)
                case onnx.AttributeProto.INTS:
                    # check if all elements are the same
                    if len(set(list(x.ints))) == 1:
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
                    assert False

        shapes = {}
        for value_info in self.onnx_model.graph.value_info:
            shape_lsts = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            shapes[value_info.name] = 'x'.join([str(x) for x in shape_lsts])
        for output in self.onnx_model.graph.output:
            shape_lsts = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            shapes[output.name] = 'x'.join([str(x) for x in shape_lsts])
        inputs = [input.name for input in self.onnx_model.graph.input]
        params = [init.name for init in self.onnx_model.graph.initializer]

        name_map = {}

        i = 1
        for input in self.onnx_model.graph.input:
            name_map[input.name] = f"{'x'.join([str(x.dim_value) for x in input.type.tensor_type.shape.dim])}"
            i += 1
        i = 1
        for init in self.onnx_model.graph.initializer:
            name_map[init.name] = f"Param{str(init.dims).replace(' ', '')}"
            i += 1
        i = 1
        for out in self.onnx_model.graph.output:
            name_map[out.name] = "Out"
            i += 1
        i = 1

        curr_str = ""
        curr_output = None
        for node in self.onnx_model.graph.node:
            node_inputs = [i for i in node.input if i not in inputs and i not in params]
            other_inputs = [name_map[i] for i in node.input if i in inputs or i in params]
            #print(node.name)
            attr = ','.join([f'{x.name}={attrtype_to_str(x)}' for x in node.attribute if attrtype_to_str(x) != '[]'])
            op_type = node.op_type
            if len(node_inputs) == 1 and len(node.output) == 1:
                # elegible for chain consecutive nodes
                if curr_output is None:
                    curr_str += f"{op_type}"
                    if attr != '':
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]
                elif curr_output in node_inputs:
                    # chain with previous node
                    curr_str += f"{op_type}"
                    if attr != '':
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]
                else:
                    # cannot chain
                    name_map[curr_output] = f"Value" # name the previous output
                    i += 1
                    if curr_output in shapes:
                        curr_str += f"{name_map[curr_output]}\n"
                    else:
                        curr_str += f"{name_map[curr_output]}\n"
                    curr_str += f"{op_type}"
                    if attr != '':
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]

            else:
                # not elegible for chaining, print current output if exists
                if curr_output is not None:
                    name_map[curr_output] = f"Value"
                    curr_str += f"{name_map[curr_output]}"
                    i += 1
                    curr_str += '\n'

                curr_str += f"{node.op_type}"
                if attr != '':
                    curr_str += f"({attr}) --> "
                else:
                    curr_str += " --> "

                for out in node.output:
                    if out not in name_map:
                        name_map[out] = f"Value"
                        i += 1
                curr_str += ", ".join([name_map[x] for x in node.output])
                if curr_str != "": 
                    curr_str += '\n'
                curr_output = None
            #print(curr_str)
                
        if curr_output is not None:
            curr_str += name_map[curr_output]

        return curr_str

    def get_onnx_infos_chain_slim_outshape(self) -> str:
        '''
        Chain consecutive nodes with single input and single output together
        '''
        def attrtype_to_str(x: onnx.AttributeProto) -> str:
            match x.type:
                case onnx.AttributeProto.FLOAT:
                    return str(x.floats)
                case onnx.AttributeProto.FLOATS:
                    return str(x.floats).replace(" ", "")
                case onnx.AttributeProto.INT:
                    return str(x.ints)
                case onnx.AttributeProto.INTS:
                    # check if all elements are the same
                    if len(set(list(x.ints))) == 1:
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
                    assert False

        shapes = {}
        for value_info in self.onnx_model.graph.value_info:
            shape_lsts = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            shapes[value_info.name] = 'x'.join([str(x) for x in shape_lsts])
        for output in self.onnx_model.graph.output:
            shape_lsts = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            shapes[output.name] = 'x'.join([str(x) for x in shape_lsts])
        inputs = [input.name for input in self.onnx_model.graph.input]
        params = [init.name for init in self.onnx_model.graph.initializer]

        name_map = {}

        i = 1
        for input in self.onnx_model.graph.input:
            name_map[input.name] = f"{'x'.join([str(x.dim_value) for x in input.type.tensor_type.shape.dim])}"
            i += 1
        i = 1
        for init in self.onnx_model.graph.initializer:
            name_map[init.name] = f"Param{str(init.dims).replace(' ', '')}"
            i += 1
        i = 1
        for out in self.onnx_model.graph.output:
            name_map[out.name] = "Out"
            i += 1
        i = 1

        curr_str = ""
        curr_output = None
        for node in self.onnx_model.graph.node:
            node_inputs = [i for i in node.input if i not in inputs and i not in params]
            other_inputs = [name_map[i] for i in node.input if i in inputs or i in params]
            #print(node.name)
            attr = ','.join([f'{x.name}={attrtype_to_str(x)}' for x in node.attribute if attrtype_to_str(x) != '[]'])
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
                    name_map[curr_output] = f"Value" # name the previous output
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
                    name_map[curr_output] = f"Value"
                    curr_str += f"{name_map[curr_output]}:{shapes[curr_output]}"
                    i += 1
                    curr_str += '\n'

                curr_str += f"{node.op_type}"
                curr_str += ") --> "

                for out in node.output:
                    if out not in name_map:
                        name_map[out] = f"Value"
                        i += 1
                curr_str += ", ".join([name_map[x] for x in node.output]) + f":{shapes[node.output[0]]}"
                if curr_str != "": 
                    curr_str += '\n'
                curr_output = None
            #print(curr_str)
                
        if curr_output is not None:
            curr_str += name_map[curr_output]

        return curr_str


    def get_onnx_infos_chain_nooutshape(self, unify_class=False, simp_op=False) -> str:
        '''
        Chain consecutive nodes with single input and single output together
        '''

        shapes = {}
        for value_info in self.onnx_model.graph.value_info:
            shapes[value_info.name] = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
        inputs = [input.name for input in self.onnx_model.graph.input]
        params = [init.name for init in self.onnx_model.graph.initializer]

        name_map = {}

        i = 1
        for input in self.onnx_model.graph.input:
            name_map[input.name] = f"In[{','.join([str(x.dim_value) for x in input.type.tensor_type.shape.dim])}]"
            i += 1
        i = 1
        for init in self.onnx_model.graph.initializer:
            name_map[init.name] = f"Param{str(init.dims).replace(' ', '')}"
            i += 1
        i = 1
        for out in self.onnx_model.graph.output:
            name_map[out.name] = f"Out[{','.join([str(x.dim_value) for x in out.type.tensor_type.shape.dim])}]"
            i += 1
        i = 1

        curr_str = ""
        curr_output = None
        for node in self.onnx_model.graph.node:
            node_inputs = [i for i in node.input if i not in inputs and i not in params]
            other_inputs = [name_map[i] for i in node.input if i in inputs or i in params]
            #print(node.name)
            attr, other_inputs = self._op_simplify(node, other_inputs, simp=simp_op)
            op_type = self._op_class(node) if unify_class else node.op_type
            if len(node_inputs) == 1 and len(node.output) == 1:
                # elegible for chain consecutive nodes
                if curr_output is None:
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0: # print additional inputs (Param or Input)
                        if node_inputs[0] in name_map:
                            curr_str += f"{name_map[node_inputs[0]]}, {', '.join(other_inputs)})"
                        else:
                            curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += f"{name_map[node_inputs[0]]})" if node_inputs[0] in name_map else "prev)"
                    if attr != '':
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]
                elif curr_output in node_inputs:
                    # chain with previous node
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0: # print additional inputs (Param or Input)
                        curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += "prev)"
                    if attr != '':
                        curr_str += f"({attr}) --> "
                    else:
                        curr_str += " --> "
                    curr_output = node.output[0]
                else:
                    # cannot chain
                    name_map[curr_output] = f"Value{i}" # name the previous output
                    i += 1
                    curr_str += f"{name_map[curr_output]}\n"
                    curr_str += f"{op_type}("
                    if len(other_inputs) > 0: # print additional inputs (Param or Input)
                        if node_inputs[0] in name_map:
                            curr_str += f"{name_map[node_inputs[0]]}, {', '.join(other_inputs)})"
                        else:
                            curr_str += f"prev, {', '.join(other_inputs)})"
                    else:
                        curr_str += ")"
                    if attr != '':
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
                    curr_str += '\n'

                curr_str += f"{node.op_type}("
                for input in node.input:
                    if input in name_map:
                        curr_str += f"{name_map[input]}, "
                    else:
                        curr_str += "prev, "
                curr_str = curr_str[:-2]  # remove last comma and space
                if attr != '':
                    curr_str += f")({attr}) --> "
                else:
                    curr_str += ") --> "

                for out in node.output:
                    if out not in name_map:
                        name_map[out] = f"Value{i}"
                        i += 1
                curr_str += ", ".join([name_map[x] for x in node.output])
                if curr_str != "": 
                    curr_str += '\n'
                curr_output = None
            #print(curr_str)
                
        if curr_output is not None:
            curr_str += name_map[curr_output]

        return curr_str

    def get_onnx_infos_minimum(self, incl_inputs=False, incl_out_shape=False) -> str:

        def attrtype_to_str(x: onnx.AttributeProto) -> str:
            match x.type:
                case onnx.AttributeProto.FLOAT:
                    return str(x.floats)
                case onnx.AttributeProto.FLOATS:
                    return str(x.floats).replace(" ", "")
                case onnx.AttributeProto.INT:
                    return str(x.ints)
                case onnx.AttributeProto.INTS:
                    # check if all elements are the same
                    if len(set(list(x.ints))) == 1:
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
                    assert False

        shapes = {}
        for value_info in self.onnx_model.graph.value_info:
            shapes[value_info.name] = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]

        name_map = {}

        i = 1
        for input in self.onnx_model.graph.input:
            name_map[input.name] = f"[{','.join([str(x.dim_value) for x in input.type.tensor_type.shape.dim])}]"
            i += 1
        i = 1
        for init in self.onnx_model.graph.initializer:
            name_map[init.name] = f"{str(init.dims).replace(' ', '')}"
            i += 1
        i = 1
        for out in self.onnx_model.graph.output:
            name_map[out.name] = f"[{','.join([str(x.dim_value) for x in out.type.tensor_type.shape.dim])}]"
            i += 1
        i = 1

        curr_str = ""
        for node in self.onnx_model.graph.node:
            if not incl_inputs:
                curr_str += f"{node.op_type}"
            else:
                curr_str += f"{node.op_type}("
                for input in node.input:
                    if input in name_map:
                        curr_str += f"{name_map[input]}, "
                    else:
                        curr_str += "prev, "
                curr_str = curr_str[:-2]  # remove last comma and space
                curr_str += ")"
            attr = ','.join([f'{x.name}={attrtype_to_str(x)}' for x in node.attribute if self._attrtype_to_str(x) != '[]'])
            if attr != '':
                curr_str += f"({attr})"
            if incl_out_shape:
                curr_str += " --> "
                curr_str += f"{shapes[node.output[0]]}" if node.output[0] in shapes else "[]"
            curr_str += '\n'
        return curr_str
    
    def get_onnx_infos_minimum_slim(self, incl_inputs=False, incl_out_shape=False) -> str:

        def attrtype_to_str(x: onnx.AttributeProto) -> str:
            match x.type:
                case onnx.AttributeProto.FLOAT:
                    return str(x.floats)
                case onnx.AttributeProto.FLOATS:
                    return str(x.floats).replace(" ", "")
                case onnx.AttributeProto.INT:
                    return str(x.ints)
                case onnx.AttributeProto.INTS:
                    # check if all elements are the same
                    if len(set(list(x.ints))) == 1:
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
                    assert False

        shapes = {}
        for value_info in self.onnx_model.graph.value_info:
            shape_lsts = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            shapes[value_info.name] = 'x'.join([str(x) for x in shape_lsts])

        name_map = {}

        i = 1
        for input in self.onnx_model.graph.input:
            name_map[input.name] = f"{'x'.join([str(x.dim_value) for x in input.type.tensor_type.shape.dim])}"
            i += 1
        i = 1
        for init in self.onnx_model.graph.initializer:
            name_map[init.name] = f"{str(init.dims).replace(' ', '')}"
            i += 1
        i = 1
        for out in self.onnx_model.graph.output:
            name_map[out.name] = f"{'x'.join([str(x.dim_value) for x in out.type.tensor_type.shape.dim])}"
            i += 1
        i = 1

        curr_str = ""
        for node in self.onnx_model.graph.node:
            if not incl_inputs:
                curr_str += f"{node.op_type}"
            else:
                curr_str += f"{node.op_type}("
                for input in node.input:
                    if input in name_map:
                        curr_str += f"{name_map[input]}, "
                    else:
                        curr_str += "prev, "
                curr_str = curr_str[:-2]  # remove last comma and space
                curr_str += ")"
            attr = ','.join([f'{x.name}={attrtype_to_str(x)}' for x in node.attribute if self._attrtype_to_str(x) != '[]'])
            if attr != '':
                curr_str += f"({attr})"
            if incl_out_shape:
                curr_str += " --> "
                curr_str += f"{shapes[node.output[0]]}" if node.output[0] in shapes else "[]"
            curr_str += '\n'
        return curr_str

    def get_onnx_infos_graph_opt(self) -> str:
        graph = onnx_to_graph(self.onnx_model)
        return encode_graph(graph)

    def get_onnx_str(self, mode: str = "full", **args) -> tuple[str, float, int]:
        mode_map = {
            "full": self.get_onnx_infos,
            "slim": self.get_onnx_infos_slim,
            "slim_noidentity": self.get_onnx_infos_slim_noidentity,
            "oponly": self.get_onnx_infos_oponly,
            "oponly_extended": self.get_onnx_infos_oponly_extended,
            "chain": self.get_onnx_infos_chain,
            "template": self.get_onnx_infos_template_compressed,
            "template_slim": self.get_onnx_infos_template_compressed_slim,
            "graph_opt": self.get_onnx_infos_graph_opt,
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
            onnx_nodes = '\n'.join(onnx_nodes)
        token_count = self._get_token_count(onnx_nodes, self.tokenizer)
        acc = self.onnx_model.metadata_props[0].value
        return onnx_nodes, acc, token_count
