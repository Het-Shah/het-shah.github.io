---
layout: post
title:  "System design basics"
author: Avishree Khare
categories: [ System Design, Interviews, tutorial ]
---

## [BASICS](https://www.youtube.com/watch?v=xpDnVSmNFX0&list=PLMCXHnjXnTnvo6alSjVkgxV-VH6EPyvoX)

**What exactly are we trying to do?**

Let's say you have some code that could help people. People would pay you for using this code.

**How will they access your code?**

You host your code on the internet and provide APIs for people to interact. They can send a REQUEST and receive a RESPONSE from an API.

**Where will this code be stored?**

If you have very few users, you could store it on your desktop. But when you have several users, you might need to host it on cloud.

**What is Cloud?**

It is essentially a group of servers somewhere where your code can be stored. You use services like AWS to buy some cloud storage.

**Why Cloud?**

Firstly, you would not that to buy so many machines. It is also comparatively cheaper. Also, you wouldn't have to worry about how the data is being stored, etc. You are paying AWS to maintain all this for you. You would only need to bother about the requirements of the users, etc. without thinking about storage.

**What if you have many many users?**

You would need to SCALE your systems. 

**How can you scale your systems?**

1. By making it bigger (VERTICAL SCALING)
2. By adding more systems (HORIZONTAL SCALING)

[Horizontal vs. Vertical Scaling](https://www.notion.so/e605db03395943e4b49b4217ee1242ec)

**So how does scaling happen in systems today?**

Systems employ both horizontal and vertical scaling. You keep making your device bigger initially as your users increase. Once, you reach a certain limit, start adding more servers.

So you use **multiple** (horizontal scaling) **bigger** (vertical scaling) **servers**.

## [NETFLIX](https://www.youtube.com/watch?v=x9Hrn0oNmJM&list=PLMCXHnjXnTnvo6alSjVkgxV-VH6EPyvoX&index=8)

**What kinds of videos are under consideration?**

1. Videos having different formats (high quality, low quality, etc.)
2. Videos of different resolutions (1080p, 720p, etc.)

**How does Netflix handle this?**

It stores videos in different formats and of different resolutions. (for example one copy of high quality and 1080p, another of high quality and 480p, etc.)

**How is the video uploaded and streamed?**

Videos need to be divided into chunks because you would not want to send the entire video at once to the user (takes time).

These chunks can be of equal time intervals (so 0 to 3 secs is one chunk, 3 to 6 is another chunk).

**There is a problem with this..**

What if there is a very critical scene from 2 to 4 seconds and your next chunk starts at 3 secs. The user would not like for a buffering at 3 secs.

**How to handle this?**

Netflix divides chunks based on scenes so that the flow doesn't break. So each chunk now becomes a scene and can be of variable lengths.

**There is another important point here.**

Some videos may be such that people only view parts of it (SPARSE videos) compared to other videos that are watched for a longer duration (DENSE videos). Netflix predicts which portions (chunks) of the videos to be sent based on this to provide a better user experience.

**So what happens when a video is requested by a user?**

We know that the video is sent in scene chunks, but from where?As most amazon servers are in the US, it might take a lot of time to send these chunks to a user in India.

To overcome this, Netflix has installed caches next to ISPs. These caches store content relevant to that area. So for someone in India, if he/she requests a bollywood movie, the ISP will first check the cache before approaching Netflix servers. Surprisingly, 90% of the requests are handled by these caches today.

**What happens when new content is uploaded?**

The content is divided into chunks and then sent to caches where it would be relevant. This is done at odd hours (say 4 PM) when there will not be any requests to handle, so the transfer can happen smoothly.

## GENERAL APPROACH

Ask the questions first:

1. What is the goal of this system
2. Who would be the users of this system
3. What are the key features required
4. What is the scale that we would be working on

## **BUILDING A CHAT APPLICATION**

1. Database tables: Which tables need to be present

    One-to-one chats: [https://www.cronj.com/blog/how-to-develop-chat-system-design-like-facebook-messenger/](https://www.cronj.com/blog/how-to-develop-chat-system-design-like-facebook-messenger/)

    Group chats allowed: [https://dba.stackexchange.com/questions/221721/how-to-structure-table-in-mysql-for-group-chat-application](https://dba.stackexchange.com/questions/221721/how-to-structure-table-in-mysql-for-group-chat-application)

2. OOP design: [https://massivetechinterview.blogspot.com/2015/07/design-chat-server-hello-world.html](https://massivetechinterview.blogspot.com/2015/07/design-chat-server-hello-world.html)