# Development Internal Support for macOS users

When trying to get virtual env and dependencies running, macOS users may

experience some compiler problems.

If that is the case, this guide is intended to help you get things working.

It was built based on common problems that were reported to the Rasa repository.

It is possible that your problem is not listed here yet, but try reading the

below because it might give you an insight.


## Problems with pyenv


### 1. ```pyenv install 3.7.6``` failed


If you tried the command above and it failed, you probably got something like

this:


Error example 1
```
> pyenv install 3.7.6
python-build: use openssl@1.1 from homebrew
python-build: use readline from homebrew
Downloading Python-3.7.6.tar.xz...
-> https://www.python.org/ftp/python/3.7.6/Python-3.7.6.tar.xz
Installing Python-3.7.6...
python-build: use readline from homebrew
python-build: use zlib from xcode sdk

BUILD FAILED (OS X 11.2.1 using python-build 20180424)

Inspect or clean up the working tree at /var/folders/g8/frp_8d7s2639spgcxqyd3kpw0000gn/T/python-build.20210215142044.19884
Results logged to /var/folders/g8/frp_8d7s2639spgcxqyd3kpw0000gn/T/python-build.20210215142044.19884.log

Last 10 log lines:
/usr/local/include/pthread.h:197:34: note: expanded from macro '_PTHREAD_SWIFT_IMPORTER_NULLABILITY_COMPAT'
        defined(SWIFT_CLASS_EXTRA) && (!defined(SWIFT_SDK_OVERLAY_PTHREAD_EPOCH) || (SWIFT_SDK_OVERLAY_PTHREAD_EPOCH < 1))
                                        ^
```

or even this:


Error example 2
```
[...]
./Modules/posixmodule.c:8436:15: error: implicit declaration of function 'sendfile' is invalid in C99 [-Werror,-Wimplicit-function-declaration]
        ret = sendfile(in, out, offset, &sbytes, &sf, flags);
              ^
1 error generated.
make: *** [Modules/posixmodule.o] Error 1
make: *** Waiting for unfinished jobs....
1 warning generated.
``` 


> **Possible solution**


Applying a **patch** when running the command worked for some users. You can do

it like this:


```$ pyenv install --patch 3.7.6 < <(curl -sSL https://github.com/python/cpython/commit/8ea6353.patch)```


The above should work just fine, but if life doesn't go easy on you, it's possible

that an error like the one below showed up:


Error example 3
```
Last 10 log lines:
  File "/private/var/folders/c8/05hjylz57llf63n1bhhv9n6w0000gn/T/python-build.20210804100934.72978/Python-3.7.6/Lib/ensurepip/__main__.py", line 5, in <module>
    sys.exit(ensurepip._main())
  File "/private/var/folders/c8/05hjylz57llf63n1bhhv9n6w0000gn/T/python-build.20210804100934.72978/Python-3.7.6/Lib/ensurepip/__init__.py", line 204, in _main
    default_pip=args.default_pip,
  File "/private/var/folders/c8/05hjylz57llf63n1bhhv9n6w0000gn/T/python-build.20210804100934.72978/Python-3.7.6/Lib/ensurepip/__init__.py", line 117, in _bootstrap
    return _run_pip(args + [p[0] for p in _PROJECTS], additional_paths)
  File "/private/var/folders/c8/05hjylz57llf63n1bhhv9n6w0000gn/T/python-build.20210804100934.72978/Python-3.7.6/Lib/ensurepip/__init__.py", line 27, in _run_pip
    import pip._internal
**zipimport.ZipImportError: can't decompress data; zlib not available**
make: *** [install] Error 1
```


> **More possible solutions**


- You may need to install *zlib*. You can do it with Homebrew:

```$ brew install zlib```

- It is possible that your shell is not able to find *zilib*. To help it,

exporting some flags may work:

```
$ export LDFLAGS="-L/usr/local/opt/zlib/lib"
$ export CPPFLAGS="-I/usr/local/opt/zlib/include"
$ export PKG_CONFIG_PATH="/usr/local/opt/zlib/lib/pkgconfig"
```

- Furthermore, you might need to create symbolic links regarding *zlib* so your

include path can find it:

```
$ ln -s /usr/local/Cellar/zlib/1.2.11/include/* /usr/local/include/
$ ln -s /usr/local/Cellar/zlib/1.2.11/lib/* /usr/local/lib/
```

- If you did everything above and still got errors, you might also reinstall

python with Homebrew:

```$ brew reinstall python```


After all that, **run the command again**:

```$ pyenv install --patch 3.7.6 < <(curl -sSL https://github.com/python/cpython/commit/8ea6353.patch)```


## Problems with my venv


- You may want to guarantee **creating your venv** with the right **python version**.

For example, ```python3 -m venv .venv```.

Then, activate the venv: ```$ source .venv/bin/activate```.


<hr>

## I've got no more problems !!!


Nice ! Now you can run one simple command and you'll be ready to work:


```$ make install```


<hr>

### References

- https://github.com/RasaHQ/rasa/issues/7956

- https://github.com/pyenv/pyenv/issues/1643

- https://github.com/python-pillow/Pillow/issues/1461

- https://stackoverflow.com/questions/38749403/python-no-module-named-zlib-mac-os-x-el-capitan-10-11-6

- https://github.com/pyenv/pyenv/issues/1764#issuecomment-758400362

- https://github.com/grpc/grpc/issues/24677#issuecomment-775458281
