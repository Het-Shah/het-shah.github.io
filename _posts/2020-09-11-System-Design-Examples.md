---
layout: post
title:  "System Design Examples"
author: [Het Shah]
author_webpages: [https://het-shah.github.io/]
categories: [ System Design, Interviews, Tutorial ]
comments: true
---

# WhatsApp System Design

### Key features

- Group messaging
- Read/receive feature
- Online/last seen
- Image sharing
- Chats are temporary/permanent

### One-to-one chat

- Gateways between a mobile and WA server
- Keeping a map of which user is connected to which gateway **(expensive).** Also, cause this will change a lot (updates in database not nice tough to make it consistent).
- Make the gateway dumb (just use it as a passing forward thing)
- Instead of a map use a sessions microservice that keeps track of the current sessions.
- Now, HTTP can't be used for receiving messages (sending is fine) because it is a client to server protocol. One bad way is long polling (poll every min whether you received a msg this is not realtime).
- New protocol over TCP - Websockets (good for chatting)
- One more database connected to the sessions microservice of messages to be delivered. So once you send a message you get notified the message is sent and then it will be delivered coz database is persistent.
- Now once the message is delivered to B, B has to send an acknowledgment (TCP handshake). Once B reads same thing again.

### Last seen/online

- A sends to gateway
- gateway to another microservice "Last seen" which keeps track of the users activity in a database.
- database me keep a threshold of like 10 seconds

### Group message

- additional microservice connected to sessions named "group service" which tells who all are in a group.
- cannot keep inf member limit (sending msg one by one will not be realtime then), limit it to x (WA has a limit of 250)
- group service will send a list to sessions
- Convert your message, using a parser like thrift (FB uses this internally), advantage is that it helps to be in a uniform machine/code language.
- Have a look at **consistent hashing.**


# Tinder System Design

IMP - Two ways to do system design

1. Start with ER diagram and then design the features what you can provide (sort of constraints you )
2. Start with the features and move towards the ER. 

### Key Features

- Store Profiles (Images important for any dating site) (Follow up ques - how many images per user, say 5)
- Recommend Matches? (Depends on many things - no of active users
- Note the matches that are made
- Direct messaging after matching

### Storing Profiles

- The main problem here is how are you going to store the images (there will be a lot of users and then for each user a constant multiple of images to be stored)
- Two ways -
    1. Files - 
    2. Blobs - Binary Large Objects (taught in database course in colleges GENERALLY (not in BITS))

        **BLOB :**

        `BLOB` (*Binary Large Object*) is a large object data type in the database system. `BLOB` could store a large chunk of data, document types and even media files like audio or video files. `BLOB` fields allocate space only whenever the content in the field is utilized. `BLOB` allocates spaces in Giga Bytes.

        **USAGE OF BLOB :**

        You can write a binary large object (`BLOB`) to a database as either binary or character data, depending on the type of field at your data source. To write a `BLOB` value to your database, issue the appropriate `INSERT or UPDATE` statement, and pass the `BLOB` value as an input parameter. If your `BLOB` is stored as text, such as a SQL Server text field, you can pass the `BLOB` as a string parameter. If the `BLOB` is stored in binary formats, such as a SQL Server image field, you can pass an array of type byte as a binary parameter.

        A useful link : [Storing documents as BLOB in Database - Any disadvantages ?](https://stackoverflow.com/questions/211895/storing-documents-as-blobs-in-a-database-any-disadvantages)

    - So what are the advantages of a DBMS vs File in general?
        1. Mutability - changing fields in rows is pretty easy
        2. Transaction guarantees - ACID Properties
        3. Indexes - To improve searching capabilities 
        4. Access Control - 
    - Mutability is not required coz you will be changing the image as a whole and not a few pixels so make it immutable.
    - Transaction guarantees not need coz we are not doing an atomic update (no one else will write your image)
    - Index searching is useful but can't search inside a image so we can implement this in code as well
    - Access control can be done in File system (tedious but can be done)

    - Advantages of FS -
        1. Cheaper 
        2. Faster than SQL querying 
        3. Build a CDN (content delivery network)
    - Use file URLs in a database and userID

    - Now first  thing is a gateway - the work of the gateway here would to authenticate the user using the username and token.
    - The gateway  authenticates this by forwarding these to the profiles service, if the profiles service says yes then the gateway will entertain any requests that are made like update profile etc etc.
    - After authenticating the gateway will forward to other microservices
    - Now in the profile section let's say you need to add images, you can keep the images in the same service as the profile service or create a new service which handles the images, namely the image service. Why? because you then just use these images directly instead of all the personal data of a user for other services like MASHEEN LEARNIN.
    - Images service will have two things -
        - distributed file system - DFS
        - table to store image urls and userID

### Chatting

- Some new points -
    - use XMPP protocol instead of HTTP - XMPP is peer to peer
    - use websockets/TCP
    - use sessions microservice to maintain connections and socket id