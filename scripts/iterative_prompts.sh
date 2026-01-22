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
  "can fish get thirsty?"
  "why don’t eggs tell jokes?"
  "what has a face and two hands but no arms or legs?"
  "if money doesn’t grow on trees, why do banks have branches?"
  "why is it called fast food if you still have to wait?"
  "can a square be round?"
  "why do we call it a building if it’s already built?"
  "what goes up but never comes down?"
  "why is it called a pair of pants if it’s only one?"
  "can you cry underwater?"
  "why do donuts have holes?"
  "what has a neck but no head?"
  "if you clean a vacuum, do you become the vacuum cleaner?"
  "why is the alphabet in that order?"
  "what has an eye but cannot see?"
  "why do we say heads up when we duck?"
  "can something be new and improved at the same time?"
  "what has legs but doesn’t walk?"
  "why is there an expiration date on salt?"
  "if you drop soap on the floor, is the floor clean or the soap dirty?"
  "what has a heart but no other organs?"
  "why are they called apartments if they’re stuck together?"
  "can you hear silence?"
  "what gets wetter the more it dries?"
  "why do we sing in the shower?"
  "what has teeth but can’t bite?"
  "if you fail a driving test, do you get a free lesson?"
  "why are there rings in tree trunks?"
  "what has a thumb and four fingers but isn’t alive?"
  "why does glue stick to the bottle?"
  "what has a head and a tail but no body?"
  "can you sneeze while sleeping?"
  "why do we knock on wood?"
  "what has words but never speaks?"
  "why do mirrors reverse left and right but not up and down?"
  "what can travel around the world while staying in one spot?"
  "why is there a D in fridge but not in refrigerator?"
  "what has a bottom at the top?"
  "can fire have a shadow?"
  "why do we say goodbye?"
  "what has ears but cannot hear?"
  "why do we dream?"
  "what gets bigger the more you take away?"
  "why do we yawn?"
  "what has a ring but no finger?"
  "can a shadow be lonely?"
  "why are pizzas round but come in square boxes?"
  "what has one eye but can’t see?"
  "why do we laugh?"
  "what has a bark but no bite?"
  "can something be invisible and still exist?"
  "why is there air?"
  "what has a spine but no bones?"
  "why do we blink?"
  "what has a face but no eyes?"
  "can you smell colors?"
  "why do we sleep?"
  "what has keys but no doors?"
  "why do we count sheep?"
  "what has pages but no words?"
  "can time be wasted?"
  "why do we dream in stories?"
  "what has wheels but doesn’t move?"
  "why do we ask questions?"
  "what has a mouth but never eats?"
  "can silence be loud?"
  "why does laughter spread?"
  "what has a tail but no body?"
)

for i in $(seq 1 200); do
  question="${questions[$RANDOM % ${#questions[@]}]}"

  echo "[$i/200] Asking: $question"

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
# This script sends 200 random questions to a local chat completion API and prints the responses.
