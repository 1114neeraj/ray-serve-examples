# Variables
ZIP_FILE = archive.zip
FILES = *.py  # Current directory

# Default target
all: zip

# Zip target
zip:
	@echo "Zipping content of $(DIR) into $(ZIP_FILE)..."
	zip -r $(ZIP_FILE) $(FILES)

# Clean target
clean:
	@echo "Cleaning up zip file..."
	rm -f $(ZIP_FILE)

serve_config:
	@echo "Creating the serve config..."
	serve build image_classifier:app translator:app -o config.yaml

deploy:
	@echo "Deploying the application..."
	serve deploy serve_config.yaml -a http://127.0.0.1:8265

# Phony targets
.PHONY: all zip clean