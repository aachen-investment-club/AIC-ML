import pytest

import alpha_generator_tests
import feature_registry_tests
import signal_generator_tests
import test_data_generator


if __name__ == "__main__":
    # Run all tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])