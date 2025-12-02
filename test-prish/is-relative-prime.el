(defun is-relative-prime (list)
  (cond
    ((null list) t)
    ((null (cdr list)) t)
    ((check-all (car list) (cdr list)) nil)
    (t (is-relative-prime (cdr list)))))