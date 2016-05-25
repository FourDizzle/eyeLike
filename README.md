##eyeLike
An OpenCV based webcam gaze tracker based on a simple image gradient-based eye center algorithm by Fabian Timm and the C++ implementation at trishume/eyeLike by Tristan Hume. 

##DISCLAIMER
This is not fully finished yet. Let me know if you're interested in helping.

##Build and Platform Dependence
To avoid having to install/compile OpenCV and to increase ease of cross platform compatibility, this uses the 3rd party OpenCV wrapper for Java, JavaCV. Because this ultimately depends on a binary compiled from C++, one of the depedencies requires the platform to be specified in a classifier. Currently, the pom is setup to automatically populate that based on the platform which is building the source. If you're aiming to build on one platform, but plan on deploying on a different you need to edit the space between the `<classifier></classifier>` tag. The options are: `android-arm`, `android-x86`, `linux-x86`, `linux-x86_64`, `macosx-x86_64`, `windows-x86`, and `windows-x86_64`.
