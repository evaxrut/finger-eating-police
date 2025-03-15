import pygame

pygame.mixer.init()
alarm_playing = False

def handle_chewing():
    global alarm_playing
    print("Chewing detected!")
    if not alarm_playing:
        pygame.mixer.music.load("alarm.wav")
        pygame.mixer.music.play(-1)
        alarm_playing = True

def stop_alarm():
    global alarm_playing
    if alarm_playing:
        pygame.mixer.music.stop()
        alarm_playing = False