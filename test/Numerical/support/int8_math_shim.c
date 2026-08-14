float floorf(float value) {
  int truncated = (int)value;
  return value < (float)truncated ? (float)(truncated - 1) : (float)truncated;
}

float ceilf(float value) {
  int truncated = (int)value;
  return value > (float)truncated ? (float)(truncated + 1) : (float)truncated;
}
