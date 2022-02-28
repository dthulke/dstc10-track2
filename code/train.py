from i6_dstc10.main import RunMode, main

import better_exchook
better_exchook.install()

if __name__ == "__main__":
    main(RunMode.TRAIN)
