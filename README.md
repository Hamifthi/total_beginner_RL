Alright so this info is based off of a handful of sources and meant to help with the general task of learning Reinforcement Learning.

Source 1: Video Lecture series and slides:
https://www.youtube.com/watch?v=lfHX2hHRMVQ&index=2&list=PL7-jPKtc4r78-wCZcQn5IqyuWhBZ8fOxT

Source 2: Move_37 RL Course with Siraj Rival:
https://www.theschool.ai/courses/

Source 3: Denny Britz's Reinforcement Learning repository with problems and solutions.
(All of the initial code from here)
https://github.com/dennybritz/reinforcement-learning

Source 4: Reinforcement Learning: An Introduction by Sutton and Barto
http://incompleteideas.net/book/bookdraft2018jan1.pdf


Policy_copy_1.py is meant to be for the absolute beginner. I've added a bunch of
print statements and code annotations, and manipulated the code so that a user can
run this script in terminal or from a python shell.

To run from terminal, first make sure you have cloned my repository so you have a copy of the "lib" file from Denny Britz's Github repository, and that you have Policy_copy_1 saved in the same directory.

It's important to have the lib file from Denny Britz because this has the code that
creates the GridWorld environment, a 4 x 4 grid which our agent is acting, "learning", in. Also his repository has links to most of the information/tutorials above and his repository
is definitely the most valuable information for learning and practicing RL. He's a boss.

Okay so with your files in order, use your terminal shell to CD into the folder with the
python script in it and then type/paste:
    python Policy_copy_1.py

The program will run, and you will see all of the data involved in the process of the
policy evaluation algorithm, so that you can explore the data to better understand
what exactly is going on, iteratively throughout the whole evaluation.
