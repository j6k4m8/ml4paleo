# Running an ml4paleo v2 site

Admins see **Administration** in the menu under their name. Admin accounts
must use two-factor sign-in.

## Who can sign up

Sign-up is open by default: anyone can make an account, with a password of at
least 12 characters that isn't a common one. Sign-ups and sign-ins are
rate-limited. To let people in only by invitation, set **Who can make an
account** to "People with an invite link", then make a link for each person
(it works once, for two weeks; with email set up and an address given, the
link is mailed too).

When email is set up, new accounts confirm their address before they can do
anything, and people can reset forgotten passwords themselves.

## Storage and models

Each account can store 10 GB and keep 20 trained models by default (change
the defaults with `M4P_QUOTA__STORAGE_GB` and `M4P_QUOTA__TRAINED_MODELS`).
A project's storage counts against its owner, including its exports while
they're kept. People ask for more from their account page; requests appear
under **Requests for more** (and are emailed to admins when email is set up).
Granting one raises that person's limits; you can also set anyone's limits
under **Accounts** → **Limits** (leave a limit empty for the default, or type
"unlimited").

## Accounts

**Accounts** lists everyone with what they use. **Disable** signs a person out
at once and stops them signing in; **Enable** lets them back in (and counts as
confirming their email). You can't disable your own account.

From the server, for an account locked out:

```sh
docker compose exec api ml4paleo-server reset-password ada       # prints a temporary password
docker compose exec api ml4paleo-server reset-two-factor ada     # they set it up again next sign-in
```

## Workers and jobs

**Workers** lists the machines that run jobs, whether they're online, and what
they're running. Workers on the same machine as the site come with it; to add
one elsewhere (say, a lab machine with a GPU), name it and make its token (see
[workers.md](workers.md)). **Revoke** stops a worker taking jobs at once; jobs
it was running go back in the queue.

**Jobs** lists recent jobs by status, with their errors, and can cancel
waiting or running ones. People see their own projects' jobs as progress on
each page.

## v1 jobs

If you import jobs from v1 (see [install.md](install.md)), the first person to
claim a job gets it. If someone claims a job that isn't theirs, release it
under **v1 jobs**: that deletes their project made from it, and the job's
owner can then claim it from its old link.
