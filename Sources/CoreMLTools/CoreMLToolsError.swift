import Foundation

public enum CoreMLToolsError: Error {
    case invalidShape
    case invalidPackagePath
    case manifestMissing
    case itemAlreadyExists(name: String, author: String)
    case rootModelAlreadyExists
}
