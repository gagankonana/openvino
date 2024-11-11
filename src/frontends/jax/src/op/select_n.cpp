// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/frontend/jax/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {

using namespace ov::op;

OutputVector translate_select_n(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    Output<Node> input = context.get_input(0);
    auto cases = context.const_named_param<std::vector<std::vector<int64_t>>>("cases");
    auto which = context.const_named_param<std::vector<int64_t>>("which");


    Output<Node> res = std::make_shared<v1::Transpose>(input, permutation_node);
    return {res};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov