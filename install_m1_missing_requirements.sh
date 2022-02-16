echo "Installing missing requirements"
for PACKAGE in $(pip check | grep -oP "(?<=requires).*(?=,)"); do
    if [ "$PACKAGE" != "tensorflow-io-gcs-filesystem" ]
    then
      VERSION=$(poetry show "$PACKAGE" | grep "version      :" | cut -d ":" -f2 | sed 's/ //g')
      pip install "$PACKAGE==$VERSION"
    fi
done
