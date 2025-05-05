

def map_range(value, in_min, in_max, out_min, out_max):
    """
    Maps a value from one range to another.

    Args:
        value (float): Input value to map.
        in_min (float): Minimum of the input range.
        in_max (float): Maximum of the input range.
        out_min (float): Minimum of the output range.
        out_max (float): Maximum of the output range.

    Returns:
        float: Mapped value in the output range.
    """
    # Clamp input value to input range
    value = max(min(value, in_max), in_min)
    
    # Perform linear mapping
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min