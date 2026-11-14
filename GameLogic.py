
import random


game_score = 501
turn_score = 0
darts_thrown = 0

while darts_thrown <3:
    throw_score = random.randint(0, 60)
    turn_score += throw_score
    darts_thrown += 1



    print("Dart", darts_thrown, "Score:", throw_score)

game_score -= turn_score

print("Turn Score:", turn_score)
print("Score:", game_score)