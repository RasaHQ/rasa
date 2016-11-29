class Trainer(object):
    SUPPORTED_LANGUAGES = None

    def ensure_language_support(self, language_name):
        if language_name not in self.SUPPORTED_LANGUAGES:
            raise NotImplementedError("MITIE backend currently does not support language '{}' (only '{}')."
                                      .format(language_name, "', '".join(self.SUPPORTED_LANGUAGES)))
