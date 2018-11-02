from setuptools import setup

# Get the long description from the README file
with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
  "rasa-core",
  "rasa-nlu",
]

setup(
    name="rasa",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        # supported python versions
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries",
    ],
    version="1.0.0",
    install_requires=install_requires,
    description="Rasa Stack - A package which includes Rasa Core and Rasa NLU",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Rasa Technologies GmbH",
    author_email="hi@rasa.com",
    maintainer="Tom Bocklisch",
    maintainer_email="tom@rasa.com",
    license="Apache 2.0",
    keywords="nlp machine-learning machine-learning-library bot bots "
             "botkit rasa conversational-agents conversational-ai chatbot"
             "chatbot-framework bot-framework",
    url="https://rasa.com",
    project_urls={
        "Bug Reports": "https://github.com/rasahq/rasa_stack/issues",
        "Source": "https://github.com/rasahq/rasa_stack",
    },
)

print("\nWelcome to Rasa!")
print("If any questions please visit documentation "
      "page https://rasa.com/docs/")
print("or join the community discussions on https://forum.rasa.com")