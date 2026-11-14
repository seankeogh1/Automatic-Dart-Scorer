game_score = 501
turn_score = 0
darts_thrown = 0

while darts_thrown <3:
    throw_score = int(input("Enter score for dart throw: "))

    turn_score += throw_score
    darts_thrown += 1

    print("Current turn score:", turn_score)

game_score -= turn_score

print("Score:", game_score)