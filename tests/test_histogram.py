"""
Tests for histogram operations.
"""
import pytest
import numpy as np
import cv2
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestHistogramOperations:
    """Test suite for histogram functions."""
    
    @pytest.fixture
    def sample_grayscale_image(self):
        """Create a sample grayscale image for testing."""
        # Create a 64x64 grayscale image with gradient
        img = np.zeros((64, 64), dtype=np.uint8)
        for i in range(64):
            img[i, :] = (i * 4) % 256  # Gradient pattern
        return img

    @pytest.fixture
    def sample_color_image(self):
        """Create a sample color image for testing."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        # Create RGB gradient
        for i in range(64):
            img[i, :, 0] = (i * 4) % 256  # Red
            img[i, :, 1] = (i * 2) % 256  # Green
            img[i, :, 2] = (i * 3) % 256  # Blue
        return img

    @pytest.fixture
    def low_contrast_image(self):
        """Create a low contrast image for equalization testing."""
        # Image with narrow histogram (low contrast)
        img = np.ones((64, 64), dtype=np.uint8) * 100
        img[20:40, 20:40] = 120
        return img

    def test_histogram_computation_grayscale(self, sample_grayscale_image):
        """Test histogram computation for grayscale image."""
        hist = cv2.calcHist([sample_grayscale_image], [0], None, [256], [0, 256])
        
        assert hist is not None
        assert hist.shape == (256, 1)
        assert np.sum(hist) == sample_grayscale_image.size

    def test_histogram_computation_color(self, sample_color_image):
        """Test histogram computation for color image (each channel)."""
        for channel in range(3):
            hist = cv2.calcHist([sample_color_image], [channel], None, [256], [0, 256])
            assert hist.shape == (256, 1)
            assert np.sum(hist) == sample_color_image.shape[0] * sample_color_image.shape[1]

    def test_histogram_equalization_grayscale(self, low_contrast_image):
        """Test histogram equalization improves contrast."""
        equalized = cv2.equalizeHist(low_contrast_image)
        
        # Check that equalized image has wider value range
        original_range = np.max(low_contrast_image) - np.min(low_contrast_image)
        equalized_range = np.max(equalized) - np.min(equalized)
        
        assert equalized.shape == low_contrast_image.shape
        assert equalized_range >= original_range

    def test_histogram_equalization_preserves_shape(self, sample_grayscale_image):
        """Test equalization preserves image dimensions."""
        equalized = cv2.equalizeHist(sample_grayscale_image)
        assert equalized.shape == sample_grayscale_image.shape

    def test_histogram_equalization_output_range(self, sample_grayscale_image):
        """Test equalized image stays within valid range."""
        equalized = cv2.equalizeHist(sample_grayscale_image)
        
        assert np.min(equalized) >= 0
        assert np.max(equalized) <= 255

    def test_histogram_sum_equals_pixel_count(self, sample_grayscale_image):
        """Test histogram bins sum to total pixel count."""
        hist = cv2.calcHist([sample_grayscale_image], [0], None, [256], [0, 256])
        total_pixels = sample_grayscale_image.shape[0] * sample_grayscale_image.shape[1]
        
        assert np.sum(hist) == total_pixels

    def test_clahe_equalization(self, low_contrast_image):
        """Test CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(low_contrast_image)
        
        assert result.shape == low_contrast_image.shape
        # CLAHE should also improve contrast
        original_std = np.std(low_contrast_image)
        result_std = np.std(result)
        assert result_std >= original_std

    def test_color_histogram_equalization(self, sample_color_image):
        """Test histogram equalization for color image via HSV."""
        # Convert to HSV
        hsv = cv2.cvtColor(sample_color_image, cv2.COLOR_BGR2HSV)
        # Equalize V channel
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        # Convert back
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        assert result.shape == sample_color_image.shape
        assert result.dtype == sample_color_image.dtype
