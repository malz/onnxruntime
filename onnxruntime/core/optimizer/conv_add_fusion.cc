// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/conv_add_fusion.h"

#include "core/common/logging/logging.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status ConvAddFusion::Apply(Graph& graph, Node& node, RewriteRuleEffect& modified) const {
  auto& conv_node = node;
  auto& add_node = *graph.GetNode((*conv_node.OutputNodesBegin()).Index());  // get mutable next node
  const auto& conv_inputs = conv_node.InputDefs();
  const auto& add_inputs = add_node.InputDefs();

  const auto* conv_W_tensor_proto = graph_utils::GetConstantInitializer(graph, conv_inputs[1]->Name());
  ORT_ENFORCE(conv_W_tensor_proto);

  const auto* add_B_tensor_proto = graph_utils::GetConstantInitializer(graph, add_inputs[1]->Name());
  ORT_ENFORCE(add_B_tensor_proto);

  // Currently, fusion is only supported for float or double data type.
  if (!Initializer::IsSupportedDataType(add_B_tensor_proto) || conv_W_tensor_proto->dims_size() < 4) {
    return Status::OK();
  }

  int axis;
  if (add_B_tensor_proto->dims_size() == conv_W_tensor_proto->dims_size()) {
    // Test for broadcast add such as 1xCx1x1 for a 2D convolution.
    axis = 1;
  } else if (add_B_tensor_proto->dims_size() == conv_W_tensor_proto->dims_size() - 1) {
    // Test for broadcast add such as Cx1x1 for a 2D convolution.
    axis = 0;
  } else {
    return Status::OK();
  }

  if (add_B_tensor_proto->dims(axis) != conv_W_tensor_proto->dims(0)) {
    return Status::OK();
  }

  // The dimensions of add_B should be equal to 1 except axis dimension.
  for (int i = 0; i < add_B_tensor_proto->dims_size(); i++) {
    if (i != axis && add_B_tensor_proto->dims(i) != 1) {
      return Status::OK();
    }
  }

  if (conv_inputs.size() == 3) {
    const auto& B_input_name = conv_inputs[2]->Name();
    const auto* conv_B_tensor_proto = graph_utils::GetConstantInitializer(graph, B_input_name);
    ORT_ENFORCE(conv_B_tensor_proto);

    if (!Initializer::IsSupportedDataType(conv_B_tensor_proto) ||
        conv_B_tensor_proto->data_type() != add_B_tensor_proto->data_type() ||
        conv_B_tensor_proto->dims_size() != 1 ||
        conv_B_tensor_proto->dims(0) != conv_W_tensor_proto->dims(0)) {
      return Status::OK();
    }

    auto conv_B = std::make_unique<Initializer>(conv_B_tensor_proto);
    auto add_B = std::make_unique<Initializer>(add_B_tensor_proto);

    if (conv_B->size() != add_B->size()) {
      return Status::OK();
    }

    // Calculate new value of initializers of conv node
    conv_B->add(*add_B);

    // Create new initializers of conv
    ONNX_NAMESPACE::TensorProto new_conv_B_tensor_proto;
    conv_B->ToProto(&new_conv_B_tensor_proto);

    auto new_name = graph.GenerateNodeArgName("ConvAddFusion_B_" + B_input_name);
    new_conv_B_tensor_proto.set_name(new_name);

    conv_node.MutableInputDefs()[2] = &graph_utils::AddReplacementInitializer(graph, new_conv_B_tensor_proto);

  } else {
    // Create new tensor proto and update shape
    ONNX_NAMESPACE::TensorProto new_conv_B_tensor_proto(*add_B_tensor_proto);
    int64_t dim = conv_W_tensor_proto->dims(0);
    new_conv_B_tensor_proto.clear_dims();
    new_conv_B_tensor_proto.add_dims(dim);

    auto new_name = graph.GenerateNodeArgName("ConvAddFusion_Add_B_" + add_B_tensor_proto->name());
    new_conv_B_tensor_proto.set_name(new_name);

    NodeArg& new_add_B_node_arg = graph_utils::AddReplacementInitializer(graph, new_conv_B_tensor_proto);

    conv_node.MutableInputDefs().push_back(&new_add_B_node_arg);
    conv_node.MutableInputArgsCount()[2] = 1;
  }

  // move the output definition and edges from the add_node to the conv_node now that they're fused
  graph_utils::DisconnectNodes(graph, conv_node, add_node);
  graph_utils::MoveOutput(graph, add_node, conv_node);

  graph.RemoveNode(add_node.Index());

  modified = RewriteRuleEffect::kModifiedRestOfGraph;

  return Status::OK();
}

bool ConvAddFusion::SatisfyCondition(const Graph& graph, const Node& node) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Conv", {1}) ||
      node.GetOutputEdgesCount() != 1) {
    return false;
  }

  const auto& next_node = *node.OutputNodesBegin();
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Add", {7}) ||
      next_node.GetInputEdgesCount() != 1 ||
      // Make sure the two nodes do not span execution providers.
      next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
    return false;
  }

  // Check that the appropriate inputs to the Conv and Add nodes are constants.
  if (!graph_utils::NodeArgIsConstant(graph, *node.InputDefs()[1]) ||
      (node.InputDefs().size() == 3 && !graph_utils::NodeArgIsConstant(graph, *node.InputDefs()[2])) ||
      !graph_utils::NodeArgIsConstant(graph, *next_node.InputDefs()[1])) {
    return false;
  }

  if (graph.IsNodeOutputsInGraphOutputs(node)) {
    return false;
  }

  return true;
}

}  // namespace onnxruntime
