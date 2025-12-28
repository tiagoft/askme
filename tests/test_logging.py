"""
Test logging functionality for information gain.
"""

import os
import tempfile
from askme.rtp.metrics import calculate_entropy, calculate_information_gain
import logging


def test_logging_to_file():
    """Test that we can log IG values to a file."""
    # Create a temporary log file
    fd, log_file = tempfile.mkstemp(suffix='.log', prefix='rtp_ig_')
    os.close(fd)  # Close the file descriptor as we'll use logging to write to it
    
    # Set up logging
    logger = logging.getLogger('askme.rtp.rtp')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Simulate some splits with IG logging
    print("Testing logging to file...")
    print(f"Log file: {log_file}")
    print()
    
    # First split
    labels1 = [0, 0, 1, 1]
    left1 = [0, 0]
    right1 = [1, 1]
    ig1 = calculate_information_gain(labels1, left1, right1)
    entropy1 = calculate_entropy(labels1)
    
    message1 = (f"Depth 0: Information Gain = {ig1:.4f} "
               f"(Entropy before: {entropy1:.4f}, "
               f"Left: {len(left1)}/{len(labels1)}, "
               f"Right: {len(right1)}/{len(labels1)})")
    print(message1)
    logger.info(message1)
    
    # Second split
    labels2 = [0, 0, 0, 1, 1, 1]
    left2 = [0, 0, 0]
    right2 = [1, 1, 1]
    ig2 = calculate_information_gain(labels2, left2, right2)
    entropy2 = calculate_entropy(labels2)
    
    message2 = (f"Depth 1: Information Gain = {ig2:.4f} "
               f"(Entropy before: {entropy2:.4f}, "
               f"Left: {len(left2)}/{len(labels2)}, "
               f"Right: {len(right2)}/{len(labels2)})")
    print(message2)
    logger.info(message2)
    
    # Third split (partial improvement)
    labels3 = [0, 0, 1, 1, 1, 1]
    left3 = [0, 0, 1]
    right3 = [1, 1, 1]
    ig3 = calculate_information_gain(labels3, left3, right3)
    entropy3 = calculate_entropy(labels3)
    
    message3 = (f"Depth 1: Information Gain = {ig3:.4f} "
               f"(Entropy before: {entropy3:.4f}, "
               f"Left: {len(left3)}/{len(labels3)}, "
               f"Right: {len(right3)}/{len(labels3)})")
    print(message3)
    logger.info(message3)
    
    # Close the handler
    file_handler.close()
    logger.removeHandler(file_handler)
    
    # Read and display the log file
    print()
    print("=" * 70)
    print("Log file contents:")
    print("=" * 70)
    with open(log_file, 'r') as f:
        content = f.read()
        print(content)
    
    # Verify log file has content
    assert os.path.exists(log_file)
    assert os.path.getsize(log_file) > 0
    
    # Clean up
    os.remove(log_file)
    print(f"\nLog file removed: {log_file}")
    print("Test passed!")


if __name__ == "__main__":
    test_logging_to_file()
