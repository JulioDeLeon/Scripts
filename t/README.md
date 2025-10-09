# GF Test Suite

This directory contains the comprehensive test suite for the gf file search tool.

## Test Structure

### Test Files

- **00-load.t** - Module loading tests
  - Verifies all modules can be loaded correctly
  - Tests basic import functionality

- **01-help-system.t** - Help system unit tests
  - Tests `show_usage()` function output
  - Tests `show_help()` function completeness
  - Tests `show_version()` function format
  - Validates help content structure and examples

- **02-error-handling.t** - Error handling unit tests
  - Tests configuration file error handling
  - Tests invalid configuration directives
  - Tests file permission error messages
  - Validates error message quality and helpfulness

- **03-integration-help.t** - Help system integration tests
  - Tests help flags work correctly with main script
  - Tests error scenarios trigger appropriate help
  - Verifies examples in documentation work correctly
  - End-to-end testing of help functionality

### Test Utilities

- **test_utils.pl** - Common testing utilities and fixtures
  - `create_test_directory()` - Creates temporary test directory structure
  - `create_test_config()` - Creates temporary configuration files
  - `capture_output()` - Captures function output for testing
  - `capture_stderr()` - Captures STDERR output for testing

- **fixtures/** - Test fixture files
  - `sample_config.conf` - Valid configuration file for testing
  - `invalid_config.conf` - Invalid configuration file for testing

## Running Tests

### Run All Tests
```bash
perl t/run_tests.pl
```

### Run Individual Tests
```bash
perl t/00-load.t
perl t/01-help-system.t
perl t/02-error-handling.t
perl t/03-integration-help.t
```

### Run with Verbose Output
```bash
perl -v t/01-help-system.t
```

## Test Coverage

The test suite covers:

### Requirements Coverage
- **1.1** - Help display when no arguments provided ✓
- **1.2** - Comprehensive help with --help flag ✓
- **1.3** - Clear error messages for invalid arguments ✓
- **1.4** - Version information display ✓
- **4.1** - Friendly error messages for missing search terms ✓
- **4.2** - Specific error information for invalid options ✓
- **4.3** - Clear context for file permission errors ✓
- **4.4** - Configuration file error explanations ✓

### Functional Coverage
- Module loading and imports
- Help system functions (usage, help, version)
- Error handling and messaging
- Configuration file processing
- Command-line argument processing
- Integration with main script
- Example validation

### Test Types
- **Unit Tests** - Individual function testing
- **Integration Tests** - End-to-end functionality testing
- **Error Handling Tests** - Error condition validation
- **Configuration Tests** - Config file processing

## Test Philosophy

The test suite follows these principles:

1. **Comprehensive Coverage** - Tests cover all major functionality and error conditions
2. **Realistic Scenarios** - Tests use realistic data and scenarios
3. **Clear Assertions** - Each test has clear, descriptive assertions
4. **Isolated Tests** - Tests don't depend on each other
5. **Fast Execution** - Tests run quickly for rapid feedback

## Adding New Tests

When adding new functionality to gf:

1. Add unit tests for new functions in appropriate test files
2. Add integration tests if the functionality affects the main script
3. Update test utilities if new common functionality is needed
4. Update this README if new test files are added

## Dependencies

The test suite uses:
- **Test::More** - Core testing framework
- **File::Temp** - Temporary file/directory creation
- **File::Path** - Directory manipulation
- **Cwd** - Directory operations

No additional CPAN modules are required beyond Perl core modules.