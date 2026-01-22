#!/bin/bash

URL="http://localhost:11436/v1/chat/completions"
MODEL="GLM-4.6V-Flash.gguf"

questions=(
  "what is the difference between a dog and a cat?"
  "why is the sky blue?"
  "if a tomato is a fruit, is ketchup a smoothie?"
  "what weighs more, a pound of feathers or a pound of bricks?"
  "why don’t skeletons fight each other?"
  "what has keys but can’t open locks?"
  "what runs but never walks?"
  "why do we park on driveways and drive on parkways?"
  "what has hands but can’t clap?"
  "why is water wet?"
  "if time flies, why do clocks exist?"
  "why do cows have best friends?"
  "what comes once in a minute, twice in a moment, but never in a thousand years?"
  "why do we press buttons harder when they don’t work?"
  "is cereal a soup?"
)

for i in $(seq 1 100); do
  question="${questions[$RANDOM % ${#questions[@]}]}"

  echo "[$i/100] Asking: $question"

  curl -s "$URL" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [
        {\"role\": \"user\", \"content\": \"$question\"}
      ]
    }" | python3 -m json.tool

  echo "----------------------------------------"
  sleep 0.1
done
# This script sends 100 random questions to a local chat completion API and prints the responses.
