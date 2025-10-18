import random

def tik_tok_to():
    print("Welcome to Tik-Tok-To!")
    print("I will think of a number between 1 and 100.")
    print("You have to guess the number.")
    print("After each guess, I will tell you if your guess is higher or lower than the number I am thinking of.")

    number_to_guess = random.randint(1, 100)
    attempts = 0

    while True:
        user_guess = input("Guess a number between 1 and 100: ")
        
        if user_guess.isdigit():
            user_guess = int(user_guess)
            attempts += 1
            
            if user_guess < number_to_guess:
                print("Your guess is too low!")
            elif user_guess > number_to_guess:
                print("Your guess is too high!")
            else:
                print(f"Congratulations! You found the number in {attempts} attempts.")
                break
        else:
            print("Invalid input. Please enter a number.")

tik_tok_to()
