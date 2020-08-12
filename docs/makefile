JOBS ?= 1

clean:
	rm -rf docs/build
	rm -rf docs/.docusaurus

install-docs:
	yarn install

build-docs:
	poetry run yarn pre-build && yarn build

livedocs:
	poetry run yarn start
