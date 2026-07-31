def test_core_types_are_available_from_package_root() -> None:
    from logqbit import LogCatalog, LogFolder, LogMetadata, LogRecord, Registry

    assert LogFolder.__name__ == "LogFolder"
    assert LogCatalog.__name__ == "LogCatalog"
    assert LogRecord.__name__ == "LogRecord"
    assert LogMetadata.__name__ == "LogMetadata"
    assert Registry.__name__ == "Registry"
