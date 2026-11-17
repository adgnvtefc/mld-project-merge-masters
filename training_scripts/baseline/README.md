# baseline
Description: the provided models and the starter solutions, scored unchanged as reference runs
Owner: both
Status: active

    # one provided checkpoint: seed, generalist, expert_math, expert_code, expert_medicine, expert_history
    ./lab run --method baseline --variant generalist --hypothesis "reference: provided generalist" \
        -- python training_scripts/baseline/provided_model.py --model generalist --out '$LAB_OUT'

    # variant names the dashboard recognizes: generalist (the Δ column's reference), expert-<domain>
    for d in math code medicine history; do
      ./lab run --method baseline --variant expert-$d --hypothesis "reference: provided $d expert" --force \
          -- python training_scripts/baseline/provided_model.py --model expert_$d --out '$LAB_OUT'
    done

    # the starter solutions
    ./lab run --method baseline --variant average-experts --hypothesis "reference: starter weight average" \
        -- python solutions/average_experts.py --out '$LAB_OUT'
    ./lab run --method baseline --variant short-training --hypothesis "reference: starter 200-step run" \
        -- python solutions/short_training.py --out '$LAB_OUT'
