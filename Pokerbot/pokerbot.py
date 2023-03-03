import random
class Pokerbot:
    def pokerhand():
        "Randomly deal 2 cards from the deck and print them."
        deck = [r+s for r in '23456789TJQKA' for s in 'SHDC']
        random.shuffle(deck)
        return deck[:2]

    def play():
        "Play a game of poker."
        hand = Pokerbot.pokerhand()
        print(hand)
        return hand

if __name__ == '__main__':
    Pokerbot.play()