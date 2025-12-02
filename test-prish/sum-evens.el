(defun sum-evens (list)
  (if (null list)
      0
      (let ((sum (sum-evens (cdr list))))
        (if (evenp (car list))
            (+ sum (car list))
            sum))))