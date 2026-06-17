//
//  FixtureLoader.swift
//  NumericSwiftWorkbenchKit
//
//  Locates and decodes the workbench JSON fixtures. Shared by the executable
//  (`NumericSwiftWorkbench`) and the XCTest gate (`WorkbenchGateTests`) so both
//  read the corpus the same way.
//
//  Licensed under the Apache License, Version 2.0.
//

import Foundation

/// Loads workbench fixtures from `Tests/NumericSwiftTests/Fixtures/workbench/`.
public enum FixtureLoader {

    /// Resolve the fixtures directory relative to this source file's location.
    ///
    /// `#filePath` → `Sources/NumericSwiftWorkbenchKit/FixtureLoader.swift`, so the
    /// package root is three levels up. Falls back to the current working
    /// directory if that path does not exist (e.g. installed builds).
    public static func fixturesDirectory() -> URL? {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // FixtureLoader.swift → NumericSwiftWorkbenchKit
            .deletingLastPathComponent()   // NumericSwiftWorkbenchKit → Sources
            .deletingLastPathComponent()   // Sources → package root

        let candidates = [
            packageRoot,
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
        ]
        for root in candidates {
            let dir = root
                .appendingPathComponent("Tests/NumericSwiftTests/Fixtures/workbench", isDirectory: true)
            if FileManager.default.fileExists(atPath: dir.path) {
                return dir
            }
        }
        return nil
    }

    /// Load fixture files for the requested domains (empty ⇒ all).
    ///
    /// - Returns: domain name → cases. Unreadable / undecodable files are reported
    ///   on stderr and skipped (never silently dropped).
    public static func load(
        domains: [String] = [],
        from directory: URL
    ) -> [String: [WorkbenchCase]] {
        var result: [String: [WorkbenchCase]] = [:]
        let fm = FileManager.default

        guard let files = try? fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: .skipsHiddenFiles
        ) else { return result }

        let jsonFiles = files
            .filter { $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        let decoder = JSONDecoder()
        for file in jsonFiles {
            let domain = file.deletingPathExtension().lastPathComponent
            guard domains.isEmpty || domains.contains(domain) else { continue }
            guard let data = try? Data(contentsOf: file) else {
                FileHandle.standardError.write(Data("Warning: cannot read \(file.path)\n".utf8))
                continue
            }
            do {
                result[domain] = try decoder.decode(FixtureFile.self, from: data)
            } catch {
                FileHandle.standardError.write(
                    Data("Warning: cannot decode \(file.path): \(error)\n".utf8))
            }
        }
        return result
    }
}
