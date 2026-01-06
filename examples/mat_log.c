#define MAT_LOG_LEVEL 3

#include "mat.h"

int main() {
  MAT_LOG_INFO("Starting example");
  MAT_LOG_WARN("This is a warning");
  MAT_LOG_ERROR("This is an error");

  return 0;
}
