#!/bin/bash

ps -ef | grep "python3 webcam-". | grep -v grep | awk '{print $2}' | xargs kill
