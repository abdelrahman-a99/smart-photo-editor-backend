"""
Tests for FFT (Fast Fourier Transform) operations.
"""
import pytest
import numpy as np
import cv2
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.fft_utils import apply_fft, apply_ifft, magnitude_spectrum


class TestFFTUtils:
    """Test suite for FFT utility functions."""
    
    @pytest.fixture
    def sample_grayscale_image(self):
        """Create a sample grayscale image for testing."""
        # Create a 64x64 grayscale image with a simple pattern
        img = np.zeros((64, 64), dtype=np.uint8)
        # Add a rectangle pattern
        img[16:48, 16:48] = 255
        return img

    @pytest.fixture
    def sample_color_image(self):
        """Create a sample color image for testing."""
        # Create a 64x64 color image
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[16:48, 16:48] = [255, 128, 64]
        return img

    def test_apply_fft_grayscale(self, sample_grayscale_image):
        """Test FFT on grayscale image returns complex array."""
        result = apply_fft(sample_grayscale_image)
        assert result is not None
        assert result.shape == sample_grayscale_image.shape
        assert np.iscomplexobj(result)

    def test_apply_fft_preserves_dimensions(self, sample_grayscale_image):
        """Test FFT preserves image dimensions."""
        result = apply_fft(sample_grayscale_image)
        assert result.shape == sample_grayscale_image.shape

    def test_apply_ifft_reconstructs_image(self, sample_grayscale_image):
        """Test inverse FFT reconstructs the original image."""
        fft_result = apply_fft(sample_grayscale_image)
        reconstructed = apply_ifft(fft_result)
        
        # Convert to same dtype for comparison
        reconstructed = np.real(reconstructed).astype(np.uint8)
        
        # Allow small numerical differences
        assert reconstructed.shape == sample_grayscale_image.shape
        np.testing.assert_allclose(
            reconstructed, 
            sample_grayscale_image, 
            atol=5  # Allow 5 pixel value tolerance
        )

    def test_magnitude_spectrum_non_negative(self, sample_grayscale_image):
        """Test magnitude spectrum returns non-negative values."""
        fft_result = apply_fft(sample_grayscale_image)
        magnitude = magnitude_spectrum(fft_result)
        
        assert magnitude is not None
        assert np.all(magnitude >= 0)

    def test_magnitude_spectrum_preserves_shape(self, sample_grayscale_image):
        """Test magnitude spectrum preserves image shape."""
        fft_result = apply_fft(sample_grayscale_image)
        magnitude = magnitude_spectrum(fft_result)
        
        assert magnitude.shape == sample_grayscale_image.shape

    def test_fft_with_different_sizes(self):
        """Test FFT works with different image sizes."""
        sizes = [(32, 32), (64, 64), (128, 128), (64, 128)]
        
        for size in sizes:
            img = np.random.randint(0, 256, size, dtype=np.uint8)
            result = apply_fft(img)
            assert result.shape == size, f"FFT failed for size {size}"
