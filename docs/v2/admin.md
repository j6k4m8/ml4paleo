# Running an ml4paleo v2 site

Admins see **Administration** in the menu under their name. Admin accounts
must use two-factor sign-in.

## Who can sign up

Sign-up is open by default: anyone can make an account, with a password of at
least 12 characters that isn't a common one. Sign-ups and sign-ins are
rate-limited. To let people in only by invitation, set **Who can make an
account** to "People with an invite link", then make a link for each person.
A link works once, for two weeks. With email set up, giving an address mails
the link to it (and signing up with it confirms that address); otherwise,
send the link yourself.

When email is set up, new accounts confirm their address before they can do
anything, and people can reset forgotten passwords themselves.

## Storage and models

Each account can store 10 GB and keep 20 trained models by default (see
[install.md](install.md) to change the defaults). A project's storage counts
against its owner, including its exports while they're kept. People ask for
more from their account page; requests appear under **Requests for more**,
and admins with an email address get them by email too (give the built-in
`admin` account one with `ml4paleo-server set-email`). Granting a request
sets the limits you fill in and leaves the others as they are; you can also
set anyone's limits under **Accounts** → **Limits** (leave a limit empty for
the default, or type "unlimited").

## Accounts

**Accounts** lists the newest accounts, or those whose username or email
starts with what you search for, with what they use and may use. **Disable**
signs a person out at once, stops the jobs they started, and keeps them from
signing in; **Enable** lets them back in (an address they never confirmed
still needs confirming). You can't disable your own account, but admins can
disable each other, which is how you'd contain a misused admin account.

From the server, for an account locked out:

```sh
docker compose exec api ml4paleo-server reset-password ada       # prints a temporary password
docker compose exec api ml4paleo-server reset-two-factor ada     # they set it up again next sign-in
docker compose exec api ml4paleo-server enable-user ada          # undo Disable
```

## Workers and jobs

**Workers** lists the machines that run jobs, whether they're online, and what
they're running. Workers on the same machine as the site come with it; to add
one elsewhere (say, a lab machine with a big GPU or lots of memory), name it
and make its token (see [workers.md](workers.md)). **Revoke** stops a worker
taking jobs at once; jobs it was running go back in the queue.

**Jobs** lists recent jobs by status, with their errors, and can cancel
waiting or running ones (with the rest of their pipeline). People see their
own projects' jobs as progress on each page.

## v1 jobs

If you import jobs from v1 (see [install.md](install.md)), the first person to
claim a job gets it. If someone claims a job that isn't theirs, give it to its
owner under **v1 jobs** (the job's id or link, and the owner's username): that
stops and deletes the project anyone else made from it, and only the owner's
account can claim it next, from its old link. Giving a job to someone again
undoes a mistake.
