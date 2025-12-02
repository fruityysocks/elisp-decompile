(defun check-all (a list)
  (cond
    ((null list) nil)
    ((eql (mygcd a (car list)) 1)
     (check-all a (cdr list)))
    (t t)))

