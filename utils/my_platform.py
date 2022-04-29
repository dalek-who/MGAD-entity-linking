import platform
import enum


class PlatformName(enum.Enum):
    TCT: str = 'Linux-4.14.105-1-tlinux3-0013-x86_64-with-centos-7.2-Final'
    MY_PC: str = 'Darwin-18.7.0-x86_64-i386-64bit'
    LAB: str = "Linux-3.10.0-1160.15.2.el7.x86_64-x86_64-with-centos-7.6.1810-Core"


dc = {
    PlatformName.TCT.value: "tct/data",
    PlatformName.MY_PC.value: "mypc/data",
    PlatformName.LAB.value: "lab/data"
}