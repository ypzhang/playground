#pragma once
#include <iostream>

namespace gs {
  /*! save and restore iostream state */
  class IosFlagSaver {
  public:
    explicit IosFlagSaver(std::ostream& _ios):
      ios(_ios),
      f(_ios.flags()) {
    }
    ~IosFlagSaver() {
      ios.flags(f);
    }

    IosFlagSaver(const IosFlagSaver &rhs) = delete;
    IosFlagSaver& operator= (const IosFlagSaver& rhs) = delete;

  private:
    std::ostream& ios;
    std::ios::fmtflags f;
  };

  double get_cur_time();

}
