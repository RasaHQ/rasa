#!/bin/bash

# MongoDB TLS Certificate Renewal Script
# This script regenerates all certificates while maintaining the exact filenames

set -e  # Exit on any error

# Configuration
CERT_DIR="./tls"
VALIDITY_DAYS=365
COUNTRY="US"
STATE="State"
CITY="City"
ORG="Organization"
OU="IT"

# Docker Configuration
# Change this to match your MongoDB container name
CONTAINER_NAME="mongodb-with-tls"
COMMON_NAME="$CONTAINER_NAME"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}MongoDB TLS Certificate Renewal Script${NC}"
echo "========================================"
echo "Container Name: $CONTAINER_NAME"
echo "(Edit CONTAINER_NAME variable in script if different)"
echo ""

# Create tls directory if it doesn't exist
mkdir -p "$CERT_DIR"
cd "$CERT_DIR"

echo -e "\n${GREEN}Step 1: Generating Certificate Authority (CA)${NC}"
# Generate CA private key
openssl genrsa -out ca.key 4096

# Generate CA certificate
openssl req -new -x509 -days $VALIDITY_DAYS -key ca.key -out ca.crt \
    -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORG/OU=$OU/CN=MongoDB-CA"

echo -e "${GREEN}✓ CA certificate generated${NC}"

echo -e "\n${GREEN}Step 2: Generating MongoDB Server Certificate${NC}"
# Generate MongoDB server private key
openssl genrsa -out mongodb.key 4096

# Generate MongoDB server certificate signing request
openssl req -new -key mongodb.key -out mongodb.csr \
    -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORG/OU=$OU/CN=$COMMON_NAME"

# Initialize serial number file
echo "1000" > ca.srl

# Create extension file for SAN (Subject Alternative Names)
cat > mongodb_ext.cnf <<EOF
subjectAltName = @alt_names
extendedKeyUsage = serverAuth,clientAuth
basicConstraints = CA:FALSE

[alt_names]
DNS.1 = $CONTAINER_NAME
DNS.2 = localhost
DNS.3 = 127.0.0.1
IP.1 = 127.0.0.1
EOF

# Sign the MongoDB server certificate with CA
openssl x509 -req -in mongodb.csr -CA ca.crt -CAkey ca.key \
    -CAcreateserial -out mongodb.crt -days $VALIDITY_DAYS \
    -extfile mongodb_ext.cnf

# Create PEM file (certificate + key combined)
cat mongodb.crt mongodb.key > mongodb.pem

# Clean up temporary files
rm mongodb.csr mongodb_ext.cnf

echo -e "${GREEN}✓ MongoDB server certificate generated${NC}"

echo -e "\n${GREEN}Step 3: Verifying Certificates${NC}"
# Verify the certificate
openssl verify -CAfile ca.crt mongodb.crt

echo -e "\n${GREEN}Certificate Details:${NC}"
echo "-------------------"
openssl x509 -in ca.crt -noout -subject -dates
echo ""
openssl x509 -in mongodb.crt -noout -subject -dates

echo -e "\n${GREEN}Step 4: Setting Permissions${NC}"
# Set appropriate permissions
chmod 600 ./*.key
chmod 644 ./*.crt ./*.pem ca.srl 2>/dev/null || true

echo -e "\n${GREEN}✓ All certificates renewed successfully!${NC}"
echo -e "\nGenerated files:"
ls -lh ca.crt ca.key ca.srl mongodb.crt mongodb.csr mongodb.key mongodb.pem 2>/dev/null || ls -lh

echo -e "\n${YELLOW}Important Notes:${NC}"
echo "1. Container name in config: $CONTAINER_NAME"
echo "2. If your MongoDB container has a different name, edit CONTAINER_NAME at the top of this script"
echo "3. Backup your old certificates before replacing them"
echo "4. Mount certificates to MongoDB container:"
echo "   docker run -v \$(pwd)/tls:/etc/mongodb/tls:ro ..."
echo "5. MongoDB command: --tlsMode requireTLS --tlsCertificateKeyFile /etc/mongodb/tls/mongodb.pem --tlsCAFile /etc/mongodb/tls/ca.crt"
echo "6. Update your application's trust store with the new ca.crt"
echo "7. Connection string: mongodb://$CONTAINER_NAME:27017/dbname?tls=true&tlsCAFile=./tls/ca.crt"
echo "8. Certificates are valid for $VALIDITY_DAYS days"
echo -e "\n${GREEN}Done!${NC}"
