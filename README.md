traceX: A 2D Ray Tracer Using CUDA
===================

This is a simple bare-bones 2D ray tracer using CUDA. I hope that I can find the time to polish it and expand it. Maybe make it Object-Oriented and cleanup the code, and tweak it to be able to do ray tracing in D dimensions (I'd really like to do ray tracing in D=26) and add some fancy features.

Info
-----
The initial problem I wanted to do, was to send a set of rays inside a mirror in the shape of a triangle. (Initially I did it with Java and you can view it in another rep "traceJ") And this is what this code is set up to, but you can change it.
The input consists of a set of boundaries and initial rays. And in different threads, and in a parallel manner the rays are traced. (thus it is much faster than traceJ and really makes use of the GPU's parallel architecture.) Each ray is R+tV where t is a scalar and R and V are 2D vectors which lie in Rx, Ry, Vx, Vy and these are dynamically updated as the ray is traced. The boundaries are similar and their R and V is stored in the same array, right after the rays. Boundaries also have an array keeping their length, because as a boundary is a line segment, it ends at a maximum t which is it's length. And to check collision with a boundary, we must solve a system of equations and find the t's which they collide with and now the t for boundary must be more than zero and less than its length, and for the ray it must be positive (a ray travels FORWARD).  During each collision the data is updated.  
I have not gone through everything here, but I hope, you can understand what happens by looking at the code.
