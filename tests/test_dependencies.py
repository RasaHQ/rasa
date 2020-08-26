import sys
import pkg_resources


def test_tensorflow_text_install():
    installed_packages_list = [i.key for i in pkg_resources.working_set]
    tf_text_installed = "tensorflow-text" in installed_packages_list

    if sys.platform == "win32":
        assert not tf_text_installed
    else:
        assert tf_text_installed
