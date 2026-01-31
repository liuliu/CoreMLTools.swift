import Foundation

public struct MILArgument {
    public enum Binding {
        case name(String)
        case value(CoreML_Specification_MILSpec_Value)
    }

    public var bindings: [Binding]

    public init(_ binding: Binding) {
        self.bindings = [binding]
    }

    public init(_ bindings: [Binding]) {
        self.bindings = bindings
    }

    public func toProto() -> CoreML_Specification_MILSpec_Argument {
        var arg = CoreML_Specification_MILSpec_Argument()
        arg.arguments = bindings.map { binding in
            var proto = CoreML_Specification_MILSpec_Argument.Binding()
            switch binding {
            case .name(let name):
                proto.name = name
            case .value(let value):
                proto.value = value
            }
            return proto
        }
        return arg
    }
}
