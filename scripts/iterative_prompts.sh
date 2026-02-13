#!/bin/bash

read -rp "Enter number of iterations [default: 129]: " iterations
iterations=${iterations:-129}

if ! [[ "$iterations" =~ ^[0-9]+$ ]] || (( iterations <= 0 )); then
  echo "Error: iterations must be a positive integer"
  exit 1
fi

URL="http://localhost:11436/v1/chat/completions"

FORMAT_PROMPT=$'You must respond with ONLY a valid JSON object.\n\
Do not include explanations, markdown, or extra text.\n\
The entire response must be JSON.\n\n\
{\n\
  "candidate_answer": "<short answer>",\n\
  "reasoning_hint": "<1–2 sentence intuition, no full explanation>",\n\
  "common_trap": "<likely wrong interpretation>",\n\
  "confidence": "low | medium | high"\n\
}\n\n\
Question: '

# ---------------- TEACHER MODE ----------------
read -rp "Enable TEACHER_MODE? [y/N]: " teacher_input
teacher_input=${teacher_input:-N}

if [[ "$teacher_input" =~ ^[Yy]$ ]]; then
  TEACHER_MODE=true
else
  TEACHER_MODE=false
fi

# ---------------- QUESTION SET SELECTION ----------------
echo "Select question set:"
echo "  1) Riddles & Curiosities"
echo "  2) Rational and Critical Thinking"
echo "  3) All Questions"
read -rp "Choice [1/2/3, default: 3]: " set_choice
set_choice=${set_choice:-3}

rational=(
  # Conceptual & Reasoning
  "Explain a complex idea using only simple words."
  "What assumptions are most people making without realizing it?"
  "Describe a problem that gets worse the harder you try to fix it."
  "What is something that looks efficient but actually isn’t?"
  "What does it mean for a system to be stable?"
  "How can two people be right and still disagree?"
  "What is a tradeoff people often ignore?"
  "When does optimization make things worse?"
  "What makes a decision reversible or irreversible?"
  "What is an example of a false choice?"
  # Planning & Strategy
  "How would you prioritize tasks with no deadlines?"
  "What is a good strategy when information is incomplete?"
  "How do you decide when to stop improving something?"
  "What’s a sign that a plan is over-engineered?"
  "How do you design something that will be used incorrectly?"
  "What’s the difference between a goal and a constraint?"
  "How do you reduce risk without slowing progress?"
  "What makes a strategy robust to failure?"
  # Ethics & Values (Non-political)
  "When is it acceptable to break a rule?"
  "What is the difference between fairness and equality?"
  "Can something be legal but still wrong?"
  "When does helping someone make them weaker?"
  "What responsibilities come with having more knowledge?"
  "Is it possible to be neutral in all situations?"
  "What makes an action well-intentioned but harmful?"
  # Systems & Complexity
  "What is an example of a feedback loop in everyday life?"
  "How do small changes lead to large outcomes?"
  "What makes a system fragile?"
  "When does redundancy improve reliability?"
  "How can local optimizations harm global performance?"
  "What does it mean for something to scale?"
  "Why do simple rules sometimes create complex behavior?"
  # Creativity & Abstraction
  "Invent a tool that solves a problem you can’t see."
  "Describe an idea without naming it."
  "What would a world optimized for comfort look like?"
  "What is something everyone learns but no one is taught?"
  "What does progress mean without measurement?"
  "How would you explain intuition to a machine?"
  "What is a question that changes its answer over time?"
  # Meta-Thinking
  "How do you know when you understand something?"
  "What is the difference between knowing and believing?"
  "What makes a question good?"
  "When does more information reduce clarity?"
  "How do you detect your own blind spots?"
  "What is the cost of being wrong?"
  "What does it mean to think clearly?"
)

riddles=(
  # Riddles & Curiosities
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

case "$set_choice" in
  1)
    questions=("${rational[@]}")
    ;;
  2)
    questions=("${riddles[@]}")
    ;;
  3)
    questions=("${rational[@]}" "${riddles[@]}")
    ;;
  *)
    echo "Invalid choice, defaulting to BOTH sets."
    questions=("${rational[@]}" "${riddles[@]}")
    ;;
esac

# ---------------- SELECTION MODE ----------------
echo "Select question order:"
echo "  1) Sequential"
echo "  2) Random"
read -rp "Choice [1/2, default: 1]: " order_choice
order_choice=${order_choice:-1}

if [[ "$order_choice" == "2" ]]; then
  RANDOM_MODE=true
else
  RANDOM_MODE=false
fi

# ---------------- SIGNAL HANDLING ----------------
stop_requested=0

stty -echoctl
cleanup() {
  stty echoctl
}
trap cleanup EXIT

sigint_handler='
if [[ $stop_requested -eq 0 ]]; then
  echo -e "\nCtrl+C detected — will exit after current iteration."
  stop_requested=1
fi
'
trap "$sigint_handler" SIGINT

# ---------------- MAIN LOOP ----------------
for ((i=1; i<=iterations; i++)); do
  if [[ $stop_requested -eq 1 ]]; then
    echo "Graceful shutdown complete. Exiting."
    break
  fi

  if ((${#questions[@]} > 0)); then
    if $RANDOM_MODE; then
      question_index=$(( RANDOM % ${#questions[@]} ))
    else
      question_index=$(( (i - 1) % ${#questions[@]} ))
    fi
    question="${questions[$question_index]}"
  else
    question="(no question supplied)"
  fi

  if $TEACHER_MODE; then
    prompt="${FORMAT_PROMPT}${question}"
  else
    prompt="$question"
  fi

  echo "[$i/$iterations] Asking: $question"

  (
    trap '' SIGINT

    jq -n \
      --arg content "$prompt" \
      '{
        messages: [
          { role: "user", content: $content }
        ]
      }' | curl -s "$URL" \
        -H "Content-Type: application/json" \
        -d @- | python3 -m json.tool
  ) &

  pid=$!

  while kill -0 "$pid" 2>/dev/null; do
    wait "$pid" 2>/dev/null || true
  done

  echo "----------------------------------------"
  sleep 0.1
done
