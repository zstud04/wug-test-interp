#!/bin/bash


if [ -f .env ]; then
    echo "Loading HF_TOKEN from .env file..."
    source .env
else
    echo "No .env file found. Please enter your HF token."
    read -p "Enter your HF_TOKEN: " HF_TOKEN

    cat <<EOF > .env
export HF_TOKEN="$HF_TOKEN"
EOF

    echo ".env file created. Token will be loaded automatically next time."
fi

echo "HF_TOKEN is set."


ENV_YML='./environment.yml'

if ! command -v conda &> /dev/null; then
    echo "Conda could not be found. Please install Conda and retry."
    exit 1
fi

if [ -f "$ENV_YML" ]; then
    ENV_NAME=$(grep "^name:" "$ENV_YML" | head -n1 | cut -d " " -f 2)
    echo "Environment name from $ENV_YML: $ENV_NAME"

    if [ "$CONDA_DEFAULT_ENV" = "$ENV_NAME" ]; then
        echo "Environment '$ENV_NAME' is already activated. Skipping creation/activation."
    else
        if conda env list | grep -qE "^[^#]*$ENV_NAME(\s|$)"; then
            echo "Environment '$ENV_NAME' already exists. Activating it..."
            conda activate "$ENV_NAME"
        else
            echo "Environment '$ENV_NAME' does not exist. Creating it from $ENV_YML..."
            conda env create -f "$ENV_YML"
            echo "Activating environment '$ENV_NAME'..."
            conda activate "$ENV_NAME"
        fi
    fi
else
    echo "No environment.yml file found at $ENV_YML"
    exit 1
fi

echo "Installing and registering R kernel in Jupyter..."
R -q -e "IRkernel::installspec(name = 'r-${ENV_NAME}', displayname = 'R (${ENV_NAME})')"

echo "Logging in to Hugging Face with your token..."
huggingface-cli login --token "$HF_TOKEN"


echo "Setup complete. Environment '$ENV_NAME' is ready for Python and R notebooks."
