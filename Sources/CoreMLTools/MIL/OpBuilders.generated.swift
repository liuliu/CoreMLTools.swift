import Foundation

public enum MILElementwiseOps {
}

public enum MILOps {
    public static func _const_symbolic(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "_const_symbolic", inputs: inputs, outputs: outputs)
    }

    public static func abs(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "abs", inputs: inputs, outputs: outputs)
    }

    public static func acos(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "acos", inputs: inputs, outputs: outputs)
    }

    public static func add(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "add", inputs: inputs, outputs: outputs)
    }

    public static func argsort(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "argsort", inputs: inputs, outputs: outputs)
    }

    public static func asin(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "asin", inputs: inputs, outputs: outputs)
    }

    public static func atan(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "atan", inputs: inputs, outputs: outputs)
    }

    public static func atanh(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "atanh", inputs: inputs, outputs: outputs)
    }

    public static func avg_pool(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "avg_pool", inputs: inputs, outputs: outputs)
    }

    public static func band_part(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "band_part", inputs: inputs, outputs: outputs)
    }

    public static func batch_norm(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "batch_norm", inputs: inputs, outputs: outputs)
    }

    public static func batch_to_space(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "batch_to_space", inputs: inputs, outputs: outputs)
    }

    public static func cast(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "cast", inputs: inputs, outputs: outputs)
    }

    public static func ceil(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "ceil", inputs: inputs, outputs: outputs)
    }

    public static func clamped_relu(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "clamped_relu", inputs: inputs, outputs: outputs)
    }

    public static func classify(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "classify", inputs: inputs, outputs: outputs)
    }

    public static func clip(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "clip", inputs: inputs, outputs: outputs)
    }

    public static func concat(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "concat", inputs: inputs, outputs: outputs)
    }

    public static func cond(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "cond", inputs: inputs, outputs: outputs)
    }

    public static func const(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "const", inputs: inputs, outputs: outputs)
    }

    public static func conv(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "conv", inputs: inputs, outputs: outputs)
    }

    public static func conv_quantized(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "conv_quantized", inputs: inputs, outputs: outputs)
    }

    public static func conv_transpose(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "conv_transpose", inputs: inputs, outputs: outputs)
    }

    public static func cos(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "cos", inputs: inputs, outputs: outputs)
    }

    public static func cosh(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "cosh", inputs: inputs, outputs: outputs)
    }

    public static func crop(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "crop", inputs: inputs, outputs: outputs)
    }

    public static func crop_resize(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "crop_resize", inputs: inputs, outputs: outputs)
    }

    public static func cumsum(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "cumsum", inputs: inputs, outputs: outputs)
    }

    public static func depth_to_space(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "depth_to_space", inputs: inputs, outputs: outputs)
    }

    public static func einsum(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "einsum", inputs: inputs, outputs: outputs)
    }

    public static func elu(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "elu", inputs: inputs, outputs: outputs)
    }

    public static func equal(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "equal", inputs: inputs, outputs: outputs)
    }

    public static func erf(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "erf", inputs: inputs, outputs: outputs)
    }

    public static func exp(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "exp", inputs: inputs, outputs: outputs)
    }

    public static func exp2(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "exp2", inputs: inputs, outputs: outputs)
    }

    public static func expand_dims(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "expand_dims", inputs: inputs, outputs: outputs)
    }

    public static func fill(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "fill", inputs: inputs, outputs: outputs)
    }

    public static func flatten2d(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "flatten2d", inputs: inputs, outputs: outputs)
    }

    public static func floor(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "floor", inputs: inputs, outputs: outputs)
    }

    public static func floor_div(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "floor_div", inputs: inputs, outputs: outputs)
    }

    public static func gather(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "gather", inputs: inputs, outputs: outputs)
    }

    public static func gather_along_axis(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "gather_along_axis", inputs: inputs, outputs: outputs)
    }

    public static func gather_nd(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "gather_nd", inputs: inputs, outputs: outputs)
    }

    public static func gelu(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "gelu", inputs: inputs, outputs: outputs)
    }

    public static func greater(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "greater", inputs: inputs, outputs: outputs)
    }

    public static func greater_equal(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "greater_equal", inputs: inputs, outputs: outputs)
    }

    public static func gru(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "gru", inputs: inputs, outputs: outputs)
    }

    public static func identity(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "identity", inputs: inputs, outputs: outputs)
    }

    public static func instance_norm(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "instance_norm", inputs: inputs, outputs: outputs)
    }

    public static func inverse(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "inverse", inputs: inputs, outputs: outputs)
    }

    public static func l2_norm(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "l2_norm", inputs: inputs, outputs: outputs)
    }

    public static func l2_pool(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "l2_pool", inputs: inputs, outputs: outputs)
    }

    public static func layer_norm(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "layer_norm", inputs: inputs, outputs: outputs)
    }

    public static func leaky_relu(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "leaky_relu", inputs: inputs, outputs: outputs)
    }

    public static func less(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "less", inputs: inputs, outputs: outputs)
    }

    public static func less_equal(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "less_equal", inputs: inputs, outputs: outputs)
    }

    public static func linear(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "linear", inputs: inputs, outputs: outputs)
    }

    public static func linear_activation(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "linear_activation", inputs: inputs, outputs: outputs)
    }

    public static func list_gather(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "list_gather", inputs: inputs, outputs: outputs)
    }

    public static func list_length(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "list_length", inputs: inputs, outputs: outputs)
    }

    public static func list_read(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "list_read", inputs: inputs, outputs: outputs)
    }

    public static func list_scatter(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "list_scatter", inputs: inputs, outputs: outputs)
    }

    public static func list_write(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "list_write", inputs: inputs, outputs: outputs)
    }

    public static func local_response_norm(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "local_response_norm", inputs: inputs, outputs: outputs)
    }

    public static func log(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "log", inputs: inputs, outputs: outputs)
    }

    public static func logical_and(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "logical_and", inputs: inputs, outputs: outputs)
    }

    public static func logical_not(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "logical_not", inputs: inputs, outputs: outputs)
    }

    public static func logical_or(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "logical_or", inputs: inputs, outputs: outputs)
    }

    public static func logical_xor(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "logical_xor", inputs: inputs, outputs: outputs)
    }

    public static func lstm(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "lstm", inputs: inputs, outputs: outputs)
    }

    public static func make_list(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "make_list", inputs: inputs, outputs: outputs)
    }

    public static func matmul(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "matmul", inputs: inputs, outputs: outputs)
    }

    public static func max_pool(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "max_pool", inputs: inputs, outputs: outputs)
    }

    public static func maximum(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "maximum", inputs: inputs, outputs: outputs)
    }

    public static func minimum(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "minimum", inputs: inputs, outputs: outputs)
    }

    public static func mod(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "mod", inputs: inputs, outputs: outputs)
    }

    public static func mul(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "mul", inputs: inputs, outputs: outputs)
    }

    public static func non_maximum_suppression(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "non_maximum_suppression", inputs: inputs, outputs: outputs)
    }

    public static func non_zero(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "non_zero", inputs: inputs, outputs: outputs)
    }

    public static func not_equal(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "not_equal", inputs: inputs, outputs: outputs)
    }

    public static func one_hot(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "one_hot", inputs: inputs, outputs: outputs)
    }

    public static func pad(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "pad", inputs: inputs, outputs: outputs)
    }

    public static func pixel_shuffle(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "pixel_shuffle", inputs: inputs, outputs: outputs)
    }

    public static func pow(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "pow", inputs: inputs, outputs: outputs)
    }

    public static func prelu(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "prelu", inputs: inputs, outputs: outputs)
    }

    public static func random_bernoulli(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "random_bernoulli", inputs: inputs, outputs: outputs)
    }

    public static func random_categorical(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "random_categorical", inputs: inputs, outputs: outputs)
    }

    public static func random_normal(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "random_normal", inputs: inputs, outputs: outputs)
    }

    public static func random_uniform(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "random_uniform", inputs: inputs, outputs: outputs)
    }

    public static func range_1d(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "range_1d", inputs: inputs, outputs: outputs)
    }

    public static func real_div(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "real_div", inputs: inputs, outputs: outputs)
    }

    public static func reduce_argmax(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "reduce_argmax", inputs: inputs, outputs: outputs)
    }

    public static func reduce_argmin(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "reduce_argmin", inputs: inputs, outputs: outputs)
    }

    public static func reduce_l1_norm(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "reduce_l1_norm", inputs: inputs, outputs: outputs)
    }

    public static func reduce_l2_norm(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "reduce_l2_norm", inputs: inputs, outputs: outputs)
    }

    public static func reduce_log_sum(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "reduce_log_sum", inputs: inputs, outputs: outputs)
    }

    public static func reduce_log_sum_exp(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "reduce_log_sum_exp", inputs: inputs, outputs: outputs)
    }

    public static func reduce_max(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "reduce_max", inputs: inputs, outputs: outputs)
    }

    public static func reduce_mean(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "reduce_mean", inputs: inputs, outputs: outputs)
    }

    public static func reduce_min(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "reduce_min", inputs: inputs, outputs: outputs)
    }

    public static func reduce_prod(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "reduce_prod", inputs: inputs, outputs: outputs)
    }

    public static func reduce_sum(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "reduce_sum", inputs: inputs, outputs: outputs)
    }

    public static func reduce_sum_square(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "reduce_sum_square", inputs: inputs, outputs: outputs)
    }

    public static func relu(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "relu", inputs: inputs, outputs: outputs)
    }

    public static func relu6(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "relu6", inputs: inputs, outputs: outputs)
    }

    public static func reshape(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "reshape", inputs: inputs, outputs: outputs)
    }

    public static func resize_bilinear(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "resize_bilinear", inputs: inputs, outputs: outputs)
    }

    public static func resize_nearest_neighbor(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "resize_nearest_neighbor", inputs: inputs, outputs: outputs)
    }

    public static func reverse(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "reverse", inputs: inputs, outputs: outputs)
    }

    public static func reverse_sequence(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "reverse_sequence", inputs: inputs, outputs: outputs)
    }

    public static func rnn(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "rnn", inputs: inputs, outputs: outputs)
    }

    public static func round(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "round", inputs: inputs, outputs: outputs)
    }

    public static func rsqrt(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "rsqrt", inputs: inputs, outputs: outputs)
    }

    public static func scaled_tanh(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "scaled_tanh", inputs: inputs, outputs: outputs)
    }

    public static func scatter(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "scatter", inputs: inputs, outputs: outputs)
    }

    public static func scatter_along_axis(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "scatter_along_axis", inputs: inputs, outputs: outputs)
    }

    public static func scatter_nd(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "scatter_nd", inputs: inputs, outputs: outputs)
    }

    public static func select(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "select", inputs: inputs, outputs: outputs)
    }

    public static func shape(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "shape", inputs: inputs, outputs: outputs)
    }

    public static func sigmoid(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "sigmoid", inputs: inputs, outputs: outputs)
    }

    public static func sigmoid_hard(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "sigmoid_hard", inputs: inputs, outputs: outputs)
    }

    public static func sign(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "sign", inputs: inputs, outputs: outputs)
    }

    public static func silu(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "silu", inputs: inputs, outputs: outputs)
    }

    public static func sin(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "sin", inputs: inputs, outputs: outputs)
    }

    public static func sinh(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "sinh", inputs: inputs, outputs: outputs)
    }

    public static func slice_by_index(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "slice_by_index", inputs: inputs, outputs: outputs)
    }

    public static func slice_by_size(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "slice_by_size", inputs: inputs, outputs: outputs)
    }

    public static func sliding_windows(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "sliding_windows", inputs: inputs, outputs: outputs)
    }

    public static func softmax(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "softmax", inputs: inputs, outputs: outputs)
    }

    public static func softplus(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "softplus", inputs: inputs, outputs: outputs)
    }

    public static func softplus_parametric(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "softplus_parametric", inputs: inputs, outputs: outputs)
    }

    public static func softsign(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "softsign", inputs: inputs, outputs: outputs)
    }

    public static func space_to_batch(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "space_to_batch", inputs: inputs, outputs: outputs)
    }

    public static func space_to_depth(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "space_to_depth", inputs: inputs, outputs: outputs)
    }

    public static func split(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "split", inputs: inputs, outputs: outputs)
    }

    public static func sqrt(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "sqrt", inputs: inputs, outputs: outputs)
    }

    public static func square(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "square", inputs: inputs, outputs: outputs)
    }

    public static func squeeze(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "squeeze", inputs: inputs, outputs: outputs)
    }

    public static func stack(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "stack", inputs: inputs, outputs: outputs)
    }

    public static func sub(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "sub", inputs: inputs, outputs: outputs)
    }

    public static func tan(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "tan", inputs: inputs, outputs: outputs)
    }

    public static func tanh(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "tanh", inputs: inputs, outputs: outputs)
    }

    public static func threshold(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "threshold", inputs: inputs, outputs: outputs)
    }

    public static func thresholded_relu(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "thresholded_relu", inputs: inputs, outputs: outputs)
    }

    public static func tile(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "tile", inputs: inputs, outputs: outputs)
    }

    public static func topk(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "topk", inputs: inputs, outputs: outputs)
    }

    public static func transpose(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "transpose", inputs: inputs, outputs: outputs)
    }

    public static func upsample_bilinear(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "upsample_bilinear", inputs: inputs, outputs: outputs)
    }

    public static func upsample_nearest_neighbor(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "upsample_nearest_neighbor", inputs: inputs, outputs: outputs)
    }

    public static func while_loop(inputs: [String: MILArgument], outputs: [CoreML_Specification_MILSpec_NamedValueType]) -> CoreML_Specification_MILSpec_Operation {
        return MILBuilder.operation(type: "while_loop", inputs: inputs, outputs: outputs)
    }

}